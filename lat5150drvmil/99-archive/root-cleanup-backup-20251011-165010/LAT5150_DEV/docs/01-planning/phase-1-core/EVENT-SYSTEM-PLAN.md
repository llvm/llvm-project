# Comprehensive Event System & Logging Infrastructure Plan

## 🎯 **Overview**

A robust event system is critical for military-grade security monitoring, audit compliance, and forensic analysis. This plan outlines the implementation of a high-performance, secure event logging infrastructure integrated with kernel tracing, audit subsystems, and persistent storage.

**CRITICAL UPDATES FROM ENUMERATION:**
- **12 DSMIL devices** generate more events than expected
- **JRTC1 marker** requires special event category for training mode
- **1.8GB hidden memory** may contain secure event logs
- **144 DSMIL ACPI references** need event monitoring

## 📋 **Current State Analysis**

### ✅ **What We Have:**
- Basic pr_debug logging to kernel ring buffer
- Simple log_event() function with event types
- Event type definitions in header
- Early boot logging buffer
- Basic debugfs event viewer

### ❌ **What's Missing:**
- Proper ring buffer implementation
- Integration with kernel trace events
- Structured event format for 12 DSMIL devices
- Event filtering and priorities
- Persistent storage in hidden memory region
- Remote logging capability
- Audit subsystem integration
- Performance optimization for increased event volume
- JRTC1-specific event categories

## 🏗️ **Comprehensive Implementation Plan**

### **Phase 1: Kernel Trace Event Infrastructure**

#### 1.1 Define Trace Events
```c
/* Create trace header: dell-milspec-trace.h */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM milspec

#if !defined(_TRACE_MILSPEC_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_MILSPEC_H

#include <linux/tracepoint.h>

/* Security event trace */
TRACE_EVENT(milspec_security,
    TP_PROTO(u32 event_id, u32 severity, const char *action, u64 data),
    TP_ARGS(event_id, severity, action, data),
    
    TP_STRUCT__entry(
        __field(u32, event_id)
        __field(u32, severity)
        __string(action, action)
        __field(u64, data)
        __field(u64, timestamp)
        __field(u32, cpu)
        __field(pid_t, pid)
    ),
    
    TP_fast_assign(
        __entry->event_id = event_id;
        __entry->severity = severity;
        __assign_str(action, action);
        __entry->data = data;
        __entry->timestamp = ktime_get_ns();
        __entry->cpu = smp_processor_id();
        __entry->pid = current->pid;
    ),
    
    TP_printk("event=%u sev=%u action=%s data=0x%llx time=%llu cpu=%u pid=%d",
              __entry->event_id, __entry->severity, __get_str(action),
              __entry->data, __entry->timestamp, __entry->cpu, __entry->pid)
);

/* Mode 5 state change trace */
TRACE_EVENT(milspec_mode5,
    TP_PROTO(int old_level, int new_level, const char *trigger),
    TP_ARGS(old_level, new_level, trigger),
    
    TP_STRUCT__entry(
        __field(int, old_level)
        __field(int, new_level)
        __string(trigger, trigger)
        __array(u8, pcr_digest, 32)
    ),
    
    TP_fast_assign(
        __entry->old_level = old_level;
        __entry->new_level = new_level;
        __assign_str(trigger, trigger);
        /* Copy TPM PCR digest */
        memcpy(__entry->pcr_digest, tpm_ctx.last_digest, 32);
    ),
    
    TP_printk("mode5: %d->%d trigger=%s pcr=%*phN",
              __entry->old_level, __entry->new_level,
              __get_str(trigger), 32, __entry->pcr_digest)
);

/* Hardware access trace */
TRACE_EVENT(milspec_mmio,
    TP_PROTO(u32 offset, u32 value, bool write),
    TP_ARGS(offset, value, write),
    
    TP_STRUCT__entry(
        __field(u32, offset)
        __field(u32, value)
        __field(bool, write)
    ),
    
    TP_fast_assign(
        __entry->offset = offset;
        __entry->value = value;
        __entry->write = write;
    ),
    
    TP_printk("mmio: %s offset=0x%03x value=0x%08x",
              __entry->write ? "write" : "read",
              __entry->offset, __entry->value)
);

#endif /* _TRACE_MILSPEC_H */

#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH .
#define TRACE_INCLUDE_FILE dell-milspec-trace
#include <trace/define_trace.h>
```

#### 1.2 Ring Buffer Implementation
```c
/* Enhanced event logging with ring buffer */
struct milspec_event {
    u64 timestamp;
    u32 event_id;
    u32 severity;
    u32 data1;
    u32 data2;
    pid_t pid;
    char comm[TASK_COMM_LEN];
    u8 payload[64];  /* Variable data */
} __packed;

struct milspec_event_buffer {
    spinlock_t lock;
    struct ring_buffer *rb;
    struct ring_buffer_event *reserve;
    atomic_t dropped;
    atomic_t total;
    bool enabled;
    u32 filter_mask;
    wait_queue_head_t waitq;
};

static struct milspec_event_buffer event_buffer;

/* Initialize enhanced event buffer */
static int milspec_init_event_buffer(void)
{
    event_buffer.rb = ring_buffer_alloc(EVENT_BUFFER_SIZE, RB_FL_OVERWRITE);
    if (!event_buffer.rb) {
        pr_err("MIL-SPEC: Failed to allocate event ring buffer\n");
        return -ENOMEM;
    }
    
    spin_lock_init(&event_buffer.lock);
    init_waitqueue_head(&event_buffer.waitq);
    atomic_set(&event_buffer.dropped, 0);
    atomic_set(&event_buffer.total, 0);
    event_buffer.enabled = true;
    event_buffer.filter_mask = 0xFFFFFFFF; /* All events */
    
    pr_info("MIL-SPEC: Event buffer initialized (%lu KB)\n",
            ring_buffer_size(event_buffer.rb) / 1024);
    
    return 0;
}

/* Log event to ring buffer */
static void milspec_log_event(u32 event_id, u32 severity, 
                              u32 data1, u32 data2,
                              void *payload, size_t payload_len)
{
    struct ring_buffer_event *event;
    struct milspec_event *entry;
    size_t length;
    unsigned long flags;
    
    if (!event_buffer.enabled || !event_buffer.rb)
        return;
    
    /* Apply filter */
    if (!(BIT(event_id) & event_buffer.filter_mask))
        return;
    
    length = sizeof(*entry) + payload_len;
    
    /* Reserve space in ring buffer */
    event = ring_buffer_lock_reserve(event_buffer.rb, length);
    if (!event) {
        atomic_inc(&event_buffer.dropped);
        return;
    }
    
    /* Fill event data */
    entry = ring_buffer_event_data(event);
    entry->timestamp = ktime_get_ns();
    entry->event_id = event_id;
    entry->severity = severity;
    entry->data1 = data1;
    entry->data2 = data2;
    entry->pid = current->pid;
    strscpy(entry->comm, current->comm, sizeof(entry->comm));
    
    if (payload && payload_len) {
        memcpy(entry->payload, payload, min(payload_len, sizeof(entry->payload)));
    }
    
    /* Commit event */
    ring_buffer_unlock_commit(event_buffer.rb, event);
    
    /* Update statistics */
    atomic_inc(&event_buffer.total);
    
    /* Wake up readers */
    wake_up_interruptible(&event_buffer.waitq);
    
    /* Also emit trace event */
    trace_milspec_security(event_id, severity, "event", 
                          ((u64)data1 << 32) | data2);
}
```

### **Phase 2: Structured Event System**

#### 2.1 Event Categories and Priorities
```c
/* Event severity levels */
enum milspec_severity {
    MILSPEC_SEV_DEBUG = 0,
    MILSPEC_SEV_INFO = 1,
    MILSPEC_SEV_NOTICE = 2,
    MILSPEC_SEV_WARNING = 3,
    MILSPEC_SEV_ERROR = 4,
    MILSPEC_SEV_CRITICAL = 5,
    MILSPEC_SEV_ALERT = 6,
    MILSPEC_SEV_EMERGENCY = 7
};

/* Event categories for filtering */
enum milspec_event_category {
    MILSPEC_CAT_BOOT = BIT(0),
    MILSPEC_CAT_SECURITY = BIT(1),
    MILSPEC_CAT_MODE5 = BIT(2),
    MILSPEC_CAT_DSMIL = BIT(3),
    MILSPEC_CAT_HARDWARE = BIT(4),
    MILSPEC_CAT_CRYPTO = BIT(5),
    MILSPEC_CAT_INTRUSION = BIT(6),
    MILSPEC_CAT_AUDIT = BIT(7),
    MILSPEC_CAT_PERFORMANCE = BIT(8),
    MILSPEC_CAT_DEBUG = BIT(9)
};

/* Extended event structure */
struct milspec_event_info {
    u32 id;
    enum milspec_severity severity;
    enum milspec_event_category category;
    const char *name;
    const char *format;
    bool audit_required;
    bool tpm_measure;
};

/* Event definitions table */
static const struct milspec_event_info event_info[] = {
    {
        .id = MILSPEC_EVENT_BOOT,
        .severity = MILSPEC_SEV_INFO,
        .category = MILSPEC_CAT_BOOT,
        .name = "boot",
        .format = "Boot stage %s reached",
        .audit_required = true,
        .tpm_measure = true
    },
    {
        .id = MILSPEC_EVENT_INTRUSION,
        .severity = MILSPEC_SEV_CRITICAL,
        .category = MILSPEC_CAT_INTRUSION | MILSPEC_CAT_SECURITY,
        .name = "intrusion",
        .format = "Physical intrusion detected on %s",
        .audit_required = true,
        .tpm_measure = true
    },
    /* ... more events ... */
};
```

#### 2.2 Event Context Capture
```c
/* Capture comprehensive event context */
struct milspec_event_context {
    /* System state */
    u32 mode5_level;
    u32 dsmil_mode;
    bool intrusion_active;
    bool wipe_armed;
    
    /* Hardware state */
    u32 mmio_status;
    u32 gpio_state;
    
    /* Security context */
    uid_t uid;
    gid_t gid;
    u32 sid;  /* SELinux SID */
    u8 pcr_digest[32];
    
    /* Stack trace for critical events */
    unsigned long backtrace[8];
    int backtrace_len;
};

static void milspec_capture_context(struct milspec_event_context *ctx)
{
    /* Capture system state */
    ctx->mode5_level = milspec_state.mode5_level;
    ctx->dsmil_mode = milspec_state.dsmil_mode;
    ctx->intrusion_active = milspec_state.intrusion_detected;
    ctx->wipe_armed = milspec_state.emergency_wipe_armed;
    
    /* Hardware state */
    if (milspec_mmio_base) {
        ctx->mmio_status = milspec_read_reg(MILSPEC_REG_STATUS);
        ctx->gpio_state = 0;
        if (mode5_gpio) ctx->gpio_state |= BIT(0);
        if (intrusion_gpio) ctx->gpio_state |= BIT(1);
    }
    
    /* Security context */
    ctx->uid = current_uid().val;
    ctx->gid = current_gid().val;
    
    /* Stack trace for debugging */
    ctx->backtrace_len = stack_trace_save(ctx->backtrace, 
                                          ARRAY_SIZE(ctx->backtrace), 0);
}
```

### **Phase 3: Audit Subsystem Integration**

#### 3.1 Linux Audit Integration
```c
#include <linux/audit.h>

/* Audit record types for MIL-SPEC */
#define AUDIT_MILSPEC_MODE5     1850
#define AUDIT_MILSPEC_INTRUSION 1851
#define AUDIT_MILSPEC_WIPE      1852
#define AUDIT_MILSPEC_DSMIL     1853

/* Send event to audit subsystem */
static void milspec_audit_event(u32 event_id, int result, 
                               const char *msg, void *data)
{
    struct audit_buffer *ab;
    const struct milspec_event_info *info;
    
    /* Find event info */
    info = milspec_find_event_info(event_id);
    if (!info || !info->audit_required)
        return;
    
    /* Allocate audit buffer */
    ab = audit_log_start(audit_context(), GFP_KERNEL, 
                         AUDIT_MILSPEC_MODE5 + info->category);
    if (!ab)
        return;
    
    /* Build audit record */
    audit_log_format(ab, "milspec_event=%s result=%d ", 
                     info->name, result);
    
    /* Add event-specific data */
    switch (event_id) {
    case MILSPEC_EVENT_MODE_CHANGE:
        audit_log_format(ab, "old_mode=%d new_mode=%d ",
                         ((u32*)data)[0], ((u32*)data)[1]);
        break;
        
    case MILSPEC_EVENT_INTRUSION:
        audit_log_format(ab, "gpio=%d type=%s ",
                         ((u32*)data)[0], 
                         ((u32*)data)[1] ? "tamper" : "intrusion");
        break;
    }
    
    /* Add common fields */
    audit_log_format(ab, "pid=%d uid=%u sessionid=%u",
                     current->pid, current_uid().val,
                     audit_get_sessionid(current));
    
    /* Add subject context */
    audit_log_task_context(ab);
    
    /* Commit audit record */
    audit_log_end(ab);
}

/* Audit filtering rules */
static int milspec_audit_filter(u32 event_id)
{
    /* Always audit critical events */
    if (event_info[event_id].severity >= MILSPEC_SEV_ERROR)
        return 1;
    
    /* Check audit rules */
    return audit_filter_rules_check(current, AUDIT_FILTER_USER);
}
```

### **Phase 4: Performance Optimization**

#### 4.1 Per-CPU Event Buffers
```c
/* Per-CPU event buffers for high performance */
struct milspec_percpu_buffer {
    struct milspec_event events[PERCPU_EVENT_COUNT];
    atomic_t head;
    atomic_t tail;
    spinlock_t flush_lock;
    struct timer_list flush_timer;
};

static DEFINE_PER_CPU(struct milspec_percpu_buffer, percpu_events);

/* Log event to per-CPU buffer */
static void milspec_log_event_percpu(u32 event_id, u32 severity,
                                    u32 data1, u32 data2)
{
    struct milspec_percpu_buffer *buffer;
    struct milspec_event *event;
    unsigned int head, tail, next;
    
    /* Use per-CPU buffer for low severity events */
    if (severity < MILSPEC_SEV_WARNING) {
        buffer = this_cpu_ptr(&percpu_events);
        
        head = atomic_read(&buffer->head);
        tail = atomic_read(&buffer->tail);
        next = (head + 1) % PERCPU_EVENT_COUNT;
        
        /* Check if buffer full */
        if (next == tail) {
            /* Flush to main buffer */
            milspec_flush_percpu_events();
        }
        
        /* Add event */
        event = &buffer->events[head];
        event->timestamp = ktime_get_ns();
        event->event_id = event_id;
        event->severity = severity;
        event->data1 = data1;
        event->data2 = data2;
        
        atomic_set(&buffer->head, next);
        
        /* Schedule flush timer */
        mod_timer(&buffer->flush_timer, jiffies + HZ);
    } else {
        /* Critical events go directly to main buffer */
        milspec_log_event(event_id, severity, data1, data2, NULL, 0);
    }
}

/* Batch flush per-CPU events */
static void milspec_flush_percpu_events(void)
{
    struct milspec_percpu_buffer *buffer;
    unsigned int cpu;
    
    for_each_online_cpu(cpu) {
        buffer = per_cpu_ptr(&percpu_events, cpu);
        /* Flush events to main ring buffer */
        /* ... */
    }
}
```

#### 4.2 Event Compression
```c
/* Event compression for repeated events */
struct milspec_event_compression {
    u32 last_event_id;
    u32 repeat_count;
    ktime_t first_time;
    ktime_t last_time;
    struct hlist_node node;
};

static DEFINE_HASHTABLE(event_compression, 8);
static DEFINE_SPINLOCK(compression_lock);

/* Compress repeated events */
static bool milspec_compress_event(u32 event_id, u32 data1, u32 data2)
{
    struct milspec_event_compression *comp;
    u32 hash = hash_32(event_id ^ data1 ^ data2, 8);
    bool compressed = false;
    
    spin_lock(&compression_lock);
    
    hash_for_each_possible(event_compression, comp, node, hash) {
        if (comp->last_event_id == event_id) {
            /* Same event within time window */
            if (ktime_us_delta(ktime_get(), comp->last_time) < 1000) {
                comp->repeat_count++;
                comp->last_time = ktime_get();
                compressed = true;
                break;
            }
        }
    }
    
    spin_unlock(&compression_lock);
    
    return compressed;
}
```

### **Phase 5: Persistent Storage**

#### 5.1 Event Log Files
```c
/* Persistent event storage */
struct milspec_event_file {
    struct file *file;
    struct mutex lock;
    loff_t size;
    u32 event_count;
    bool rotating;
};

#define EVENT_LOG_PATH "/var/log/milspec/events.log"
#define EVENT_LOG_MAX_SIZE (10 * 1024 * 1024) /* 10MB */

static struct milspec_event_file event_file;

/* Write events to persistent storage */
static int milspec_write_event_file(struct milspec_event *event)
{
    char buffer[512];
    int len;
    
    if (!event_file.file)
        return -ENOENT;
    
    /* Format event */
    len = snprintf(buffer, sizeof(buffer),
        "[%llu.%06lu] %s:%s: id=%u sev=%u data=%08x:%08x pid=%d[%s]\n",
        event->timestamp / 1000000000ULL,
        (event->timestamp % 1000000000ULL) / 1000,
        current->comm,
        event_info[event->event_id].name,
        event->event_id,
        event->severity,
        event->data1,
        event->data2,
        event->pid,
        event->comm);
    
    /* Write to file */
    kernel_write(event_file.file, buffer, len, &event_file.file->f_pos);
    
    /* Check rotation */
    event_file.size += len;
    if (event_file.size > EVENT_LOG_MAX_SIZE) {
        milspec_rotate_event_log();
    }
    
    return 0;
}

/* Log rotation */
static void milspec_rotate_event_log(void)
{
    char oldpath[256], newpath[256];
    int i;
    
    /* Rotate existing logs */
    for (i = 9; i > 0; i--) {
        snprintf(oldpath, sizeof(oldpath), "%s.%d", EVENT_LOG_PATH, i - 1);
        snprintf(newpath, sizeof(newpath), "%s.%d", EVENT_LOG_PATH, i);
        kernel_rename(oldpath, newpath);
    }
    
    /* Move current to .0 */
    kernel_rename(EVENT_LOG_PATH, oldpath);
    
    /* Create new log file */
    milspec_open_event_file();
}
```

### **Phase 6: User Interface**

#### 6.1 Debugfs Event Reader
```c
/* Enhanced debugfs interface */
static int milspec_events_show(struct seq_file *m, void *v)
{
    struct ring_buffer_event *rbe;
    struct milspec_event *event;
    struct ring_buffer_iter *iter;
    u64 ts;
    
    /* Create iterator */
    iter = ring_buffer_read_prepare(event_buffer.rb, 0);
    if (!iter)
        return -ENOMEM;
    
    ring_buffer_read_start(iter);
    
    /* Read all events */
    while ((rbe = ring_buffer_iter_peek(iter, &ts))) {
        event = ring_buffer_event_data(rbe);
        
        seq_printf(m, "[%llu.%06lu] Event: %s (id=%u sev=%u)\n",
                   ts / 1000000000ULL,
                   (ts % 1000000000ULL) / 1000,
                   event_info[event->event_id].name,
                   event->event_id,
                   event->severity);
        
        seq_printf(m, "  PID: %d [%s]\n", event->pid, event->comm);
        seq_printf(m, "  Data: 0x%08x 0x%08x\n", 
                   event->data1, event->data2);
        
        if (event->payload[0]) {
            seq_printf(m, "  Payload: %*ph\n", 16, event->payload);
        }
        
        seq_puts(m, "\n");
        
        ring_buffer_read(iter, NULL);
    }
    
    ring_buffer_read_finish(iter);
    
    /* Show statistics */
    seq_printf(m, "Total events: %u\n", atomic_read(&event_buffer.total));
    seq_printf(m, "Dropped events: %u\n", atomic_read(&event_buffer.dropped));
    
    return 0;
}
```

#### 6.2 Real-time Event Monitoring
```c
/* Character device for real-time monitoring */
static ssize_t milspec_events_read(struct file *file, char __user *buf,
                                  size_t count, loff_t *ppos)
{
    struct milspec_event_reader *reader = file->private_data;
    struct ring_buffer_event *rbe;
    struct milspec_event *event;
    size_t read = 0;
    int ret;
    
    /* Wait for events */
    ret = wait_event_interruptible(event_buffer.waitq,
                                   ring_buffer_entries(event_buffer.rb) > 0 ||
                                   !event_buffer.enabled);
    if (ret)
        return ret;
    
    /* Read available events */
    while (read + sizeof(*event) <= count) {
        rbe = ring_buffer_consume(event_buffer.rb, 0, NULL);
        if (!rbe)
            break;
        
        event = ring_buffer_event_data(rbe);
        
        /* Apply reader filter */
        if (reader->filter_mask & BIT(event->event_id)) {
            if (copy_to_user(buf + read, event, sizeof(*event)))
                return -EFAULT;
            
            read += sizeof(*event);
        }
    }
    
    return read;
}

static const struct file_operations milspec_events_fops = {
    .owner = THIS_MODULE,
    .open = milspec_events_open,
    .release = milspec_events_release,
    .read = milspec_events_read,
    .poll = milspec_events_poll,
    .unlocked_ioctl = milspec_events_ioctl,
};
```

## 📊 **Implementation Priority**

### **High Priority:**
1. Ring buffer implementation
2. Basic trace event integration
3. Structured event system
4. Debugfs interface

### **Medium Priority:**
5. Audit subsystem integration
6. Per-CPU optimization
7. Event compression
8. Persistent storage

### **Low Priority:**
9. Remote logging
10. Advanced filtering
11. Event correlation
12. Graphical analysis tools

## ⚠️ **Security Considerations**

1. **Access Control**: Event logs contain sensitive security information
2. **Integrity**: Events must be tamper-proof (hash chain)
3. **Availability**: Logging must not be disruptable by attacks
4. **Confidentiality**: Encrypt sensitive event data
5. **Compliance**: Meet military audit requirements

## 📅 **Implementation Timeline**

- **Week 1**: Core ring buffer and trace events
- **Week 2**: Structured events and context capture
- **Week 3**: Audit integration and optimization
- **Week 4**: Persistent storage and user interfaces
- **Week 5**: Testing and hardening

## 🔧 **Testing Strategy**

1. **Performance Tests**: Event throughput and latency
2. **Stress Tests**: High event rate scenarios
3. **Security Tests**: Attempt to bypass or corrupt logs
4. **Integration Tests**: With audit and trace subsystems
5. **Compliance Tests**: Verify military standards

---

**Status**: Plan Ready for Implementation
**Priority**: High - Required for security compliance
**Estimated Effort**: 5 weeks development + testing
**Dependencies**: Kernel trace infrastructure, audit subsystem