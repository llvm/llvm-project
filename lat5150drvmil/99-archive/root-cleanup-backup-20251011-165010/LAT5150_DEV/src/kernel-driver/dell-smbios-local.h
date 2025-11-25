/* Local copy of dell-smbios.h for out-of-tree module building */
#ifndef _DELL_SMBIOS_LOCAL_H_
#define _DELL_SMBIOS_LOCAL_H_

#include <linux/types.h>

/* SMBIOS calling interface buffer */
struct calling_interface_buffer {
    u16 cmd_class;
    u16 cmd_select;
    u32 input[4];
    u32 output[4];
};

/* Common tokens from Dell SMBIOS */
#define DELL_SMBIOS_MODE5_TOKEN     0x04F0
#define DELL_SMBIOS_DSMIL_TOKEN     0x04F1
#define DELL_SMBIOS_MILSPEC_TOKEN   0x04F2

/* Function placeholders for out-of-tree building */
static inline int dell_smbios_call(struct calling_interface_buffer *buffer)
{
    /* In a real implementation, this would call the Dell SMBIOS interface */
    pr_warn("dell-milspec: Dell SMBIOS not available in out-of-tree build\n");
    return -ENOTSUPP;
}

#endif /* _DELL_SMBIOS_LOCAL_H_ */