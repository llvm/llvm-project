/*
 * Dell SMBIOS Token Enumeration Module
 * Safe discovery tool for DSMIL device control tokens
 * 
 * SECURITY NOTE: READ-ONLY access only, avoids dangerous token ranges
 * Dangerous ranges: 0x8000-0x8014 (security), 0xF600-0xF601 (military)
 * 
 * Copyright (C) 2025 Dell Latitude 5450 MIL-SPEC DSMIL Research Project
 * Licensed under GPL v2
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/platform_device.h>
#include <linux/device.h>
#include <linux/sysfs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/jiffies.h>

/* Dell SMBIOS interface headers */
#ifdef CONFIG_DELL_SMBIOS
#include <linux/dell-smbios.h>
#else
/* Fallback definitions if dell-smbios not available */
struct calling_interface_buffer {
	u16 class;
	u16 select;
	volatile u32 input[4];
	volatile u32 output[4];
} __packed;

#define DELL_SMBIOS_SELECT_TOKENS	1
#define DELL_SMBIOS_CMD_GET_TOKENS	0x01
#define DELL_SMBIOS_CMD_ENUM_TOKEN	0x02
#define DELL_SMBIOS_CLASS_INFO		17
#endif

#define DRIVER_NAME "dell-smbios-token-enum"
#define DRIVER_VERSION "1.0.0"
#define DRIVER_DESC "Safe Dell SMBIOS Token Enumeration for DSMIL Discovery"

/* Token enumeration parameters */
static uint max_tokens = 1024;
module_param(max_tokens, uint, 0644);
MODULE_PARM_DESC(max_tokens, "Maximum number of tokens to enumerate (default 1024)");

static bool enable_verbose = false;
module_param(enable_verbose, bool, 0644);
MODULE_PARM_DESC(enable_verbose, "Enable verbose token logging (default false)");

static bool emergency_stop = false;
module_param(emergency_stop, bool, 0644);
MODULE_PARM_DESC(emergency_stop, "Emergency stop flag - halt all operations");

static uint scan_delay_ms = 100;
module_param(scan_delay_ms, uint, 0644);
MODULE_PARM_DESC(scan_delay_ms, "Delay between token reads in milliseconds (default 100)");

/* Token type definitions */
enum dell_token_type {
	DELL_TOKEN_TYPE_UNKNOWN = 0,
	DELL_TOKEN_TYPE_BOOL = 1,
	DELL_TOKEN_TYPE_INTEGER = 2,
	DELL_TOKEN_TYPE_STRING = 3,
	DELL_TOKEN_TYPE_BINARY = 4,
	DELL_TOKEN_TYPE_ENUM = 5,
	DELL_TOKEN_TYPE_PASSWORD = 6,
	DELL_TOKEN_TYPE_KEY = 7
};

/* Token access level definitions */
enum dell_token_access {
	DELL_TOKEN_ACCESS_PUBLIC = 0,
	DELL_TOKEN_ACCESS_ADMIN = 1,
	DELL_TOKEN_ACCESS_SECURITY = 2,
	DELL_TOKEN_ACCESS_FACTORY = 3,
	DELL_TOKEN_ACCESS_RESTRICTED = 4
};

/* Token structure for discovered tokens */
struct dell_discovered_token {
	u16 token_id;
	u8 token_type;
	u8 access_level;
	u16 token_flags;
	u32 current_value;
	u32 default_value;
	u32 min_value;
	u32 max_value;
	bool is_readable;
	bool is_writable;
	bool requires_auth;
	bool is_dangerous;
	char name[64];
	char description[128];
	struct list_head list;
};

/* Safe token ranges - explicitly exclude dangerous ranges */
struct token_range {
	u16 start;
	u16 end;
	const char *description;
	bool safe_to_read;
	bool safe_to_enumerate;
};

/* Token range definitions with safety classifications */
static const struct token_range token_ranges[] = {
	/* Standard Dell system tokens - SAFE */
	{ 0x0001, 0x00FF, "Standard System Configuration", true, true },
	{ 0x0100, 0x01FF, "Hardware Configuration", true, true },
	{ 0x0200, 0x02FF, "Power Management", true, true },
	{ 0x0300, 0x03FF, "Display Configuration", true, true },
	{ 0x0400, 0x04FF, "Audio Configuration", true, true },
	{ 0x0500, 0x05FF, "Network Configuration", true, true },
	{ 0x0600, 0x06FF, "Storage Configuration", true, true },
	{ 0x0700, 0x07FF, "Thermal Management", true, true },
	
	/* Extended system tokens - SAFE */
	{ 0x1000, 0x1FFF, "Extended System Settings", true, true },
	{ 0x2000, 0x2FFF, "Application Settings", true, true },
	{ 0x3000, 0x3FFF, "User Preferences", true, true },
	{ 0x4000, 0x4FFF, "Diagnostics Settings", true, true },
	{ 0x5000, 0x5FFF, "Asset Management", true, true },
	{ 0x6000, 0x6FFF, "Service Configuration", true, true },
	{ 0x7000, 0x7FFF, "Vendor Extensions", true, true },
	
	/* DANGEROUS RANGES - DO NOT ACCESS */
	{ 0x8000, 0x8014, "MIL-SPEC Security Tokens", false, false },	/* NEVER TOUCH */
	{ 0x8015, 0x80FF, "Extended Security", false, false },		/* CAUTION */
	{ 0x8100, 0x81FF, "Operational Commands", false, false },	/* DANGEROUS */
	{ 0x8200, 0x82FF, "Hidden Memory Control", false, false },	/* RISKY */
	{ 0x8300, 0x83FF, "JRTC1 Training", true, true },		/* Training - safer */
	{ 0x8400, 0x84FF, "DSMIL Device Control", true, true },	/* TARGET RANGE */
	{ 0x8500, 0x8FFF, "Future MIL-SPEC", false, false },		/* UNKNOWN */
	
	/* High ranges - potentially dangerous */
	{ 0x9000, 0x9FFF, "Reserved Range 1", false, false },
	{ 0xA000, 0xAFFF, "Reserved Range 2", false, false },
	{ 0xB000, 0xEFFF, "Extended Reserved", false, false },
	{ 0xF000, 0xF5FF, "System Reserved", false, false },
	{ 0xF600, 0xF601, "Military Override", false, false },		/* NEVER TOUCH */
	{ 0xF602, 0xFFFF, "Factory Reserved", false, false }
};

/* Driver state structure */
struct dell_token_enum_state {
	struct platform_device *pdev;
	struct list_head discovered_tokens;
	struct mutex token_mutex;
	struct proc_dir_entry *proc_entry;
	struct kobject *sysfs_kobj;
	
	/* Statistics */
	u32 tokens_scanned;
	u32 tokens_found;
	u32 tokens_accessible;
	u32 tokens_dangerous;
	u32 ranges_scanned;
	u32 errors_encountered;
	
	/* Safety controls */
	bool enumeration_active;
	bool emergency_stopped;
	ktime_t scan_start_time;
	ktime_t last_scan_time;
	
	/* DSMIL-specific discoveries */
	u32 dsmil_tokens_found;
	u32 potential_device_tokens;
	u16 dsmil_base_token;
	bool dsmil_pattern_detected;
};

static struct dell_token_enum_state *enum_state;

/* Forward declarations */
static int dell_smbios_safe_call(struct calling_interface_buffer *buffer);
static bool is_token_safe_to_read(u16 token_id);
static bool is_token_range_safe(u16 token_id);
static int enumerate_token_range(u16 start, u16 end, const char *description);
static int discover_single_token(u16 token_id);
static void analyze_dsmil_patterns(void);
static const char *get_token_type_name(u8 type);
static const char *get_access_level_name(u8 level);

/* Safe Dell SMBIOS call wrapper with error handling */
static int dell_smbios_safe_call(struct calling_interface_buffer *buffer)
{
	int ret = -ENODEV;
	
	/* Check emergency stop */
	if (emergency_stop || enum_state->emergency_stopped) {
		pr_warn(DRIVER_NAME ": Emergency stop active, aborting SMBIOS call\n");
		return -EACCES;
	}
	
	/* Add safety delay between calls */
	if (scan_delay_ms > 0) {
		msleep(scan_delay_ms);
	}
	
#ifdef CONFIG_DELL_SMBIOS
	/* Use kernel dell-smbios if available */
	ret = dell_smbios_call(buffer);
	if (ret) {
		enum_state->errors_encountered++;
		if (enable_verbose) {
			pr_debug(DRIVER_NAME ": SMBIOS call failed with error %d\n", ret);
		}
	}
#else
	/* Fallback: simulate safe operation */
	pr_info(DRIVER_NAME ": Dell SMBIOS not available, using simulation mode\n");
	buffer->output[0] = 0;
	buffer->output[1] = 0;
	buffer->output[2] = 0;
	buffer->output[3] = 0;
	ret = 0;
#endif

	return ret;
}

/* Check if a token ID is in a safe range for reading */
static bool is_token_safe_to_read(u16 token_id)
{
	int i;
	
	for (i = 0; i < ARRAY_SIZE(token_ranges); i++) {
		const struct token_range *range = &token_ranges[i];
		
		if (token_id >= range->start && token_id <= range->end) {
			if (!range->safe_to_read) {
				pr_warn(DRIVER_NAME ": Token 0x%04X in dangerous range: %s\n", 
					token_id, range->description);
				return false;
			}
			return true;
		}
	}
	
	/* Unknown range - be conservative */
	pr_debug(DRIVER_NAME ": Token 0x%04X in unknown range, treating as unsafe\n", token_id);
	return false;
}

/* Check if a token range is safe for enumeration */
static bool is_token_range_safe(u16 token_id)
{
	int i;
	
	for (i = 0; i < ARRAY_SIZE(token_ranges); i++) {
		const struct token_range *range = &token_ranges[i];
		
		if (token_id >= range->start && token_id <= range->end) {
			return range->safe_to_enumerate;
		}
	}
	
	return false;
}

/* Discover properties of a single token */
static int discover_single_token(u16 token_id)
{
	struct calling_interface_buffer buffer = {0};
	struct dell_discovered_token *token;
	int ret;
	
	/* Safety check */
	if (!is_token_safe_to_read(token_id)) {
		enum_state->tokens_dangerous++;
		return -EACCES;
	}
	
	enum_state->tokens_scanned++;
	
	/* Try to read token information */
	buffer.class = DELL_SMBIOS_CLASS_INFO;
	buffer.select = DELL_SMBIOS_SELECT_TOKENS;
	buffer.input[0] = DELL_SMBIOS_CMD_ENUM_TOKEN;
	buffer.input[1] = token_id;
	buffer.input[2] = 0;  /* Read operation */
	buffer.input[3] = 0;
	
	ret = dell_smbios_safe_call(&buffer);
	if (ret != 0) {
		/* Token not accessible or error occurred */
		return ret;
	}
	
	/* Check if we got valid response */
	if (buffer.output[0] == 0xFFFFFFFF) {
		/* Token doesn't exist */
		return -ENOENT;
	}
	
	/* Create token record */
	token = kzalloc(sizeof(*token), GFP_KERNEL);
	if (!token) {
		return -ENOMEM;
	}
	
	/* Fill in discovered information */
	token->token_id = token_id;
	token->current_value = buffer.output[0];
	token->token_type = (buffer.output[1] >> 24) & 0xFF;
	token->token_flags = buffer.output[1] & 0xFFFF;
	token->access_level = (buffer.output[2] >> 24) & 0xFF;
	token->default_value = buffer.output[2] & 0xFFFFFF;
	token->min_value = buffer.output[3] >> 16;
	token->max_value = buffer.output[3] & 0xFFFF;
	
	/* Analyze token characteristics */
	token->is_readable = true;  /* We just read it */
	token->is_writable = !(token->token_flags & 0x01);  /* Read-only flag */
	token->requires_auth = !!(token->token_flags & 0x02);  /* Auth required flag */
	token->is_dangerous = !is_token_range_safe(token_id);
	
	/* Generate descriptive name */
	snprintf(token->name, sizeof(token->name), "Token_%04X", token_id);
	
	/* Generate description based on range */
	for (int i = 0; i < ARRAY_SIZE(token_ranges); i++) {
		const struct token_range *range = &token_ranges[i];
		if (token_id >= range->start && token_id <= range->end) {
			snprintf(token->description, sizeof(token->description),
				"%s (ID: 0x%04X)", range->description, token_id);
			break;
		}
	}
	
	/* Add to discovered tokens list */
	mutex_lock(&enum_state->token_mutex);
	list_add_tail(&token->list, &enum_state->discovered_tokens);
	enum_state->tokens_found++;
	enum_state->tokens_accessible++;
	
	/* Check for DSMIL-related patterns */
	if ((token_id >= 0x8400 && token_id <= 0x84FF) ||  /* DSMIL control range */
	    (token_id >= 0x8300 && token_id <= 0x83FF)) {  /* JRTC1 training range */
		enum_state->dsmil_tokens_found++;
		
		/* Check for patterns suggesting device control */
		if ((token->current_value & 0xFF000000) == 0x44000000) {  /* 'D' prefix */
			enum_state->potential_device_tokens++;
			if (enable_verbose) {
				pr_info(DRIVER_NAME ": Potential DSMIL device token 0x%04X: value=0x%08X\n",
					token_id, token->current_value);
			}
		}
	}
	
	mutex_unlock(&enum_state->token_mutex);
	
	if (enable_verbose) {
		pr_info(DRIVER_NAME ": Discovered token 0x%04X: type=%s, value=0x%08X, flags=0x%04X\n",
			token_id, get_token_type_name(token->token_type), 
			token->current_value, token->token_flags);
	}
	
	return 0;
}

/* Enumerate tokens in a specific range */
static int enumerate_token_range(u16 start, u16 end, const char *description)
{
	u16 token_id;
	int discovered = 0;
	int ret;
	
	pr_info(DRIVER_NAME ": Enumerating range 0x%04X-0x%04X: %s\n", 
		start, end, description);
	
	enum_state->ranges_scanned++;
	
	for (token_id = start; token_id <= end; token_id++) {
		/* Check emergency stop */
		if (emergency_stop || enum_state->emergency_stopped) {
			pr_warn(DRIVER_NAME ": Emergency stop triggered, halting enumeration\n");
			break;
		}
		
		ret = discover_single_token(token_id);
		if (ret == 0) {
			discovered++;
		}
		
		/* Throttle to prevent overwhelming the system */
		if ((token_id % 16) == 0) {
			cond_resched();
		}
	}
	
	pr_info(DRIVER_NAME ": Range 0x%04X-0x%04X complete: %d tokens discovered\n",
		start, end, discovered);
	
	return discovered;
}

/* Analyze discovered tokens for DSMIL patterns */
static void analyze_dsmil_patterns(void)
{
	struct dell_discovered_token *token;
	u32 device_pattern_count = 0;
	u32 control_pattern_count = 0;
	u32 status_pattern_count = 0;
	
	pr_info(DRIVER_NAME ": Analyzing DSMIL patterns in %d discovered tokens\n",
		enum_state->tokens_found);
	
	mutex_lock(&enum_state->token_mutex);
	
	list_for_each_entry(token, &enum_state->discovered_tokens, list) {
		/* Look for device control patterns */
		if (token->token_id >= 0x8400 && token->token_id <= 0x847F) {
			/* Potential device tokens */
			if ((token->current_value & 0xFF000000) == 0x44000000) {  /* 'D' prefix */
				device_pattern_count++;
				pr_info(DRIVER_NAME ": Device pattern 0x%04X: 0x%08X (potential DSMIL%d)\n",
					token->token_id, token->current_value,
					((token->current_value >> 16) & 0xFF));
			}
		}
		
		/* Look for control command patterns */
		if (token->token_id >= 0x8480 && token->token_id <= 0x84FF) {
			if ((token->current_value & 0xFFFF0000) == 0x43540000) {  /* 'CT' prefix */
				control_pattern_count++;
				pr_info(DRIVER_NAME ": Control pattern 0x%04X: 0x%08X\n",
					token->token_id, token->current_value);
			}
		}
		
		/* Look for status/state patterns */
		if ((token->current_value & 0xFF000000) == 0x53000000) {  /* 'S' prefix */
			status_pattern_count++;
		}
	}
	
	mutex_unlock(&enum_state->token_mutex);
	
	/* Generate analysis report */
	pr_info(DRIVER_NAME ": DSMIL Pattern Analysis Complete:\n");
	pr_info(DRIVER_NAME ":   Device patterns: %d\n", device_pattern_count);
	pr_info(DRIVER_NAME ":   Control patterns: %d\n", control_pattern_count);
	pr_info(DRIVER_NAME ":   Status patterns: %d\n", status_pattern_count);
	pr_info(DRIVER_NAME ":   Total DSMIL tokens: %d\n", enum_state->dsmil_tokens_found);
	pr_info(DRIVER_NAME ":   Potential device control: %d\n", enum_state->potential_device_tokens);
	
	if (device_pattern_count > 0) {
		pr_info(DRIVER_NAME ": DSMIL device control tokens detected - potential 72-device control mechanism found\n");
		enum_state->dsmil_pattern_detected = true;
	}
}

/* Get human-readable token type name */
static const char *get_token_type_name(u8 type)
{
	switch (type) {
	case DELL_TOKEN_TYPE_UNKNOWN:  return "Unknown";
	case DELL_TOKEN_TYPE_BOOL:     return "Boolean";
	case DELL_TOKEN_TYPE_INTEGER:  return "Integer";
	case DELL_TOKEN_TYPE_STRING:   return "String";
	case DELL_TOKEN_TYPE_BINARY:   return "Binary";
	case DELL_TOKEN_TYPE_ENUM:     return "Enumeration";
	case DELL_TOKEN_TYPE_PASSWORD: return "Password";
	case DELL_TOKEN_TYPE_KEY:      return "Cryptographic Key";
	default:                       return "Reserved";
	}
}

/* Get human-readable access level name */
static const char *get_access_level_name(u8 level)
{
	switch (level) {
	case DELL_TOKEN_ACCESS_PUBLIC:     return "Public";
	case DELL_TOKEN_ACCESS_ADMIN:      return "Administrator";
	case DELL_TOKEN_ACCESS_SECURITY:   return "Security";
	case DELL_TOKEN_ACCESS_FACTORY:    return "Factory";
	case DELL_TOKEN_ACCESS_RESTRICTED: return "Restricted";
	default:                           return "Unknown";
	}
}

/* Proc filesystem interface for token information */
static int token_enum_proc_show(struct seq_file *m, void *v)
{
	struct dell_discovered_token *token;
	ktime_t elapsed;
	
	seq_printf(m, "Dell SMBIOS Token Enumeration Report\n");
	seq_printf(m, "=====================================\n\n");
	
	/* Statistics */
	elapsed = ktime_sub(ktime_get(), enum_state->scan_start_time);
	seq_printf(m, "Scan Statistics:\n");
	seq_printf(m, "  Ranges scanned: %d\n", enum_state->ranges_scanned);
	seq_printf(m, "  Tokens scanned: %d\n", enum_state->tokens_scanned);
	seq_printf(m, "  Tokens found: %d\n", enum_state->tokens_found);
	seq_printf(m, "  Accessible tokens: %d\n", enum_state->tokens_accessible);
	seq_printf(m, "  Dangerous tokens avoided: %d\n", enum_state->tokens_dangerous);
	seq_printf(m, "  Errors encountered: %d\n", enum_state->errors_encountered);
	seq_printf(m, "  Scan time: %lld ms\n\n", ktime_to_ms(elapsed));
	
	/* DSMIL Analysis */
	seq_printf(m, "DSMIL Analysis:\n");
	seq_printf(m, "  DSMIL tokens found: %d\n", enum_state->dsmil_tokens_found);
	seq_printf(m, "  Potential device tokens: %d\n", enum_state->potential_device_tokens);
	seq_printf(m, "  DSMIL pattern detected: %s\n", 
		   enum_state->dsmil_pattern_detected ? "YES" : "NO");
	seq_printf(m, "\n");
	
	/* Token Details */
	seq_printf(m, "Discovered Tokens:\n");
	seq_printf(m, "==================\n");
	seq_printf(m, "ID     Type      Access    Value      Flags  Description\n");
	seq_printf(m, "------ --------- --------- ---------- ------ -----------\n");
	
	mutex_lock(&enum_state->token_mutex);
	list_for_each_entry(token, &enum_state->discovered_tokens, list) {
		seq_printf(m, "0x%04X %-9s %-9s 0x%08X 0x%04X %s\n",
			   token->token_id,
			   get_token_type_name(token->token_type),
			   get_access_level_name(token->access_level),
			   token->current_value,
			   token->token_flags,
			   token->description);
	}
	mutex_unlock(&enum_state->token_mutex);
	
	return 0;
}

static int token_enum_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, token_enum_proc_show, NULL);
}

static const struct proc_ops token_enum_proc_ops = {
	.proc_open = token_enum_proc_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};

/* Sysfs interface for emergency stop */
static ssize_t emergency_stop_show(struct device *dev,
				   struct device_attribute *attr, 
				   char *buf)
{
	return sprintf(buf, "%d\n", enum_state->emergency_stopped ? 1 : 0);
}

static ssize_t emergency_stop_store(struct device *dev,
				    struct device_attribute *attr,
				    const char *buf, size_t count)
{
	int value;
	
	if (kstrtoint(buf, 10, &value) != 0)
		return -EINVAL;
		
	enum_state->emergency_stopped = !!value;
	emergency_stop = !!value;
	
	if (value) {
		pr_warn(DRIVER_NAME ": Emergency stop activated by user\n");
	} else {
		pr_info(DRIVER_NAME ": Emergency stop deactivated by user\n");
	}
	
	return count;
}

static DEVICE_ATTR_RW(emergency_stop);

static struct attribute *token_enum_attrs[] = {
	&dev_attr_emergency_stop.attr,
	NULL,
};

static const struct attribute_group token_enum_attr_group = {
	.attrs = token_enum_attrs,
};

/* Platform driver probe */
static int dell_token_enum_probe(struct platform_device *pdev)
{
	int ret, i;
	
	pr_info(DRIVER_NAME ": Probing Dell SMBIOS Token Enumeration System\n");
	
	/* Initialize driver state */
	enum_state = kzalloc(sizeof(*enum_state), GFP_KERNEL);
	if (!enum_state)
		return -ENOMEM;
		
	enum_state->pdev = pdev;
	INIT_LIST_HEAD(&enum_state->discovered_tokens);
	mutex_init(&enum_state->token_mutex);
	enum_state->scan_start_time = ktime_get();
	
	platform_set_drvdata(pdev, enum_state);
	
	/* Create proc entry */
	enum_state->proc_entry = proc_create("dell-token-enum", 0444, NULL, 
					     &token_enum_proc_ops);
	if (!enum_state->proc_entry) {
		pr_warn(DRIVER_NAME ": Failed to create proc entry\n");
	}
	
	/* Create sysfs interface */
	ret = sysfs_create_group(&pdev->dev.kobj, &token_enum_attr_group);
	if (ret) {
		pr_warn(DRIVER_NAME ": Failed to create sysfs interface: %d\n", ret);
	}
	
	/* Start safe token enumeration */
	pr_info(DRIVER_NAME ": Starting SAFE token enumeration (avoiding dangerous ranges)\n");
	pr_info(DRIVER_NAME ": NEVER accessing: 0x8000-0x8014 (security), 0xF600-0xF601 (military)\n");
	
	enum_state->enumeration_active = true;
	
	/* Enumerate safe ranges only */
	for (i = 0; i < ARRAY_SIZE(token_ranges); i++) {
		const struct token_range *range = &token_ranges[i];
		
		if (!range->safe_to_enumerate) {
			pr_info(DRIVER_NAME ": Skipping dangerous range 0x%04X-0x%04X: %s\n",
				range->start, range->end, range->description);
			continue;
		}
		
		if (emergency_stop || enum_state->emergency_stopped) {
			pr_warn(DRIVER_NAME ": Emergency stop during enumeration\n");
			break;
		}
		
		enumerate_token_range(range->start, range->end, range->description);
	}
	
	enum_state->enumeration_active = false;
	enum_state->last_scan_time = ktime_get();
	
	/* Analyze discovered patterns */
	analyze_dsmil_patterns();
	
	pr_info(DRIVER_NAME ": Token enumeration complete\n");
	pr_info(DRIVER_NAME ": Found %d total tokens, %d DSMIL-related, %d potential device control\n",
		enum_state->tokens_found, enum_state->dsmil_tokens_found, 
		enum_state->potential_device_tokens);
	
	return 0;
}

/* Platform driver remove */
static void dell_token_enum_remove(struct platform_device *pdev)
{
	struct dell_discovered_token *token, *next;
	
	pr_info(DRIVER_NAME ": Removing token enumeration driver\n");
	
	/* Set emergency stop */
	enum_state->emergency_stopped = true;
	
	/* Remove sysfs interface */
	sysfs_remove_group(&pdev->dev.kobj, &token_enum_attr_group);
	
	/* Remove proc entry */
	if (enum_state->proc_entry) {
		proc_remove(enum_state->proc_entry);
	}
	
	/* Free discovered tokens */
	mutex_lock(&enum_state->token_mutex);
	list_for_each_entry_safe(token, next, &enum_state->discovered_tokens, list) {
		list_del(&token->list);
		kfree(token);
	}
	mutex_unlock(&enum_state->token_mutex);
	
	/* Free driver state */
	kfree(enum_state);
	
	pr_info(DRIVER_NAME ": Token enumeration driver removed safely\n");
}

/* Platform driver structure */
static struct platform_driver dell_token_enum_driver = {
	.driver = {
		.name = DRIVER_NAME,
		.owner = THIS_MODULE,
	},
	.probe = dell_token_enum_probe,
	.remove = dell_token_enum_remove,
};

/* Module initialization */
static int __init dell_token_enum_init(void)
{
	int ret;
	
	pr_info(DRIVER_NAME ": Initializing Dell SMBIOS Token Enumeration v%s\n", 
		DRIVER_VERSION);
	pr_info(DRIVER_NAME ": SAFETY MODE - Read-only enumeration with dangerous range avoidance\n");
	
	ret = platform_driver_register(&dell_token_enum_driver);
	if (ret) {
		pr_err(DRIVER_NAME ": Failed to register platform driver: %d\n", ret);
		return ret;
	}
	
	/* Register dummy platform device */
	platform_device_register_simple(DRIVER_NAME, -1, NULL, 0);
	
	return 0;
}

/* Module cleanup */
static void __exit dell_token_enum_exit(void)
{
	pr_info(DRIVER_NAME ": Exiting Dell SMBIOS Token Enumeration\n");
	platform_driver_unregister(&dell_token_enum_driver);
}

module_init(dell_token_enum_init);
module_exit(dell_token_enum_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Dell Latitude 5450 MIL-SPEC DSMIL Research Team");
MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_VERSION(DRIVER_VERSION);
MODULE_ALIAS("platform:" DRIVER_NAME);