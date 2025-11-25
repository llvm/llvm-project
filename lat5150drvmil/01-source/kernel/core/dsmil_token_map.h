/* Auto-generated DSMIL token map. DO NOT EDIT BY HAND. */
#ifndef DSMIL_TOKEN_MAP_H
#define DSMIL_TOKEN_MAP_H

#include <linux/types.h>

struct dsmil_static_token_map_entry {
    u32 device_id;       /* e.g. 0x8000 */
    u32 token;           /* raw token dword (LE) */
    u32 control;         /* control dword (LE) */
    u32 desc_offset;     /* descriptor page offset (0x0..0x8000) */
    u32 leaf_offset;     /* resolved leaf page (0 if unknown) */
    const char *name;    /* device name from catalog, if known */
};

static const struct dsmil_static_token_map_entry dsmil_static_token_map[] = {
    { 0x8000, 0x00800003, 0x00200000, 0x0000, 0x0000, "TPMControlDevice" },
    { 0x8010, 0x00801003, 0x00200000, 0x0000, 0x0000, "IntrusionDetectionDevice" },
    { 0x8020, 0x00802003, 0x00200000, 0x0000, 0x0000, "FrequencyHopDevice" },
    { 0x8030, 0x00803003, 0x00200000, 0x0000, 0x0000, "StorageEncryptionDevice" },
    { 0x8040, 0x00804003, 0x00200000, 0x0000, 0x0000, "HapticFeedbackDevice" },
    { 0x8050, 0x00805003, 0x00200000, 0x0000, 0x0000, "StorageEncryptionDevice" },
    { 0x8060, 0x00806003, 0x00200000, 0x0000, 0x0000, "unknown" },
};

static const u32 dsmil_static_token_map_count =
    (u32)(sizeof(dsmil_static_token_map)/sizeof(dsmil_static_token_map[0]));

#endif /* DSMIL_TOKEN_MAP_H */
