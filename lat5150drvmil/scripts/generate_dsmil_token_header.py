#!/usr/bin/env python3
"""
Generate a static header mapping core DSMIL tokens to descriptor/control info.

Reads 04-hardware/microcode/dsmil_token_summary.json (built from shadow tables)
and emits 01-source/kernel/dsmil_token_map.h with an array of entries for
token ids in the core range (0x8000–0x806B).
"""
import json
from pathlib import Path

SRC = Path('04-hardware/microcode/dsmil_token_summary.json')
OUT = Path('01-source/kernel/dsmil_token_map.h')
CAT = Path('DSMIL_DEVICE_CAPABILITIES.json')
RESOLVED = Path('04-hardware/microcode/dsmil_resolved_map.json')

def main():
    if not SRC.exists():
        raise SystemExit(f"Input JSON not found: {SRC}")
    data = json.loads(SRC.read_text())
    name_map = {}
    if CAT.exists():
        try:
            cap = json.loads(CAT.read_text())
            devices = cap.get('devices', {})
            for k,v in devices.items():
                try:
                    name_map[int(k,16)] = v.get('name') or 'unknown'
                except Exception:
                    continue
        except Exception:
            pass
    table = data.get('table', [])
    # Optional resolved leaf offsets
    leaf_map = {}
    if RESOLVED.exists():
        try:
            r = json.loads(RESOLVED.read_text())
            for entry in r.get('devices', []):
                try:
                    dev = int(entry['device_id'], 16)
                    leaf = int(entry.get('leaf_offset') or '0', 16)
                    leaf_map[dev] = leaf
                except Exception:
                    continue
        except Exception:
            pass

    # Collect first-seen mapping per core token id (0x8000..0x806B)
    core_min, core_max = 0x8000, 0x806B
    seen = {}
    for row in table:
        try:
            token_hex = row['token']
            control_hex = row['control']
            token = int(token_hex, 16)
            control = int(control_hex, 16)
            # Compute device id directly from token (bits 31..8)
            device_id = (token >> 8) & 0xffff
            # Prefer catalog name derived from device_id
            name = name_map.get(device_id, row.get('name') or 'unknown')
            desc_off = int(row.get('offset') or row.get('descriptor_offset') or 0)
        except Exception:
            continue
        if device_id < core_min or device_id > core_max:
            continue
        if device_id in seen:
            continue
        seen[device_id] = (token, control, desc_off, name)

    # Emit the header
    lines = []
    lines.append('/* Auto-generated DSMIL token map. DO NOT EDIT BY HAND. */')
    lines.append('#ifndef DSMIL_TOKEN_MAP_H')
    lines.append('#define DSMIL_TOKEN_MAP_H')
    lines.append('')
    lines.append('#include <linux/types.h>')
    lines.append('')
    lines.append('struct dsmil_static_token_map_entry {')
    lines.append('    u32 device_id;       /* e.g. 0x8000 */')
    lines.append('    u32 token;           /* raw token dword (LE) */')
    lines.append('    u32 control;         /* control dword (LE) */')
    lines.append('    u32 desc_offset;     /* descriptor page offset (0x0..0x8000) */')
    lines.append('    u32 leaf_offset;     /* resolved leaf page (0 if unknown) */')
    lines.append('    const char *name;    /* device name from catalog, if known */')
    lines.append('};')
    lines.append('')
    lines.append('static const struct dsmil_static_token_map_entry dsmil_static_token_map[] = {')
    for dev in sorted(seen.keys()):
        token, control, desc_off, name = seen[dev]
        leaf_off = leaf_map.get(dev, 0)
        # sanitize name for C string
        name_c = name.replace('"','\\"')
        lines.append(f'    {{ 0x{dev:04x}, 0x{token:08x}, 0x{control:08x}, 0x{desc_off:04x}, 0x{leaf_off:04x}, "{name_c}" }},')
    lines.append('};')
    lines.append('')
    lines.append('static const u32 dsmil_static_token_map_count =')
    lines.append('    (u32)(sizeof(dsmil_static_token_map)/sizeof(dsmil_static_token_map[0]));')
    lines.append('')
    lines.append('#endif /* DSMIL_TOKEN_MAP_H */')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {OUT} with {len(seen)} entries")

if __name__ == '__main__':
    main()
