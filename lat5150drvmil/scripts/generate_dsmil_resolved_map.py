#!/usr/bin/env python3
"""
Generate a resolved device map (including leaf offsets) by querying the
DSMIL driver's sysfs resolver for each core token (0x8000..0x806B).

Inputs:
  - 04-hardware/microcode/dsmil_token_summary.json
Outputs:
  - 04-hardware/microcode/dsmil_resolved_map.json

This script stays strictly read-only (no firmware writes). It depends on the
DSMIL kernel module being loaded and exposing '/sys/class/dsmil-*/resolve_token'.
"""
import json
import os
from pathlib import Path

SRC = Path('04-hardware/microcode/dsmil_token_summary.json')
OUT = Path('04-hardware/microcode/dsmil_resolved_map.json')


def find_sysfs_base():
    candidates = [
        Path('/sys/class/dsmil-84dev/dsmil-84dev'),
        Path('/sys/class/dsmil-72dev/dsmil-72dev'),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('DSMIL sysfs base not found')


def parse_resolve_line(s: str):
    # token=0x00800003 next=0x123 leaf=0x456 depth=2
    parts = dict()
    for tok in s.strip().split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            parts[k] = v
    return parts


def main():
    if not SRC.exists():
        raise SystemExit(f'Missing input: {SRC}')
    base = find_sysfs_base()
    resolve_path = base / 'resolve_token'
    if not resolve_path.exists():
        raise SystemExit('resolve_token sysfs attribute is missing (update driver)')

    data = json.loads(SRC.read_text())
    table = data.get('table', [])

    # Pick first-seen mapping for each core device id
    core_min, core_max = 0x8000, 0x806B
    seen = {}
    for row in table:
        try:
            token = int(row['token'], 16)
            control = int(row['control'], 16)
            device_id = (token >> 8) & 0xFFFF
            desc_off = int(row.get('offset') or row.get('descriptor_offset') or 0)
        except Exception:
            continue
        if device_id < core_min or device_id > core_max:
            continue
        if device_id in seen:
            continue
        seen[device_id] = (token, control, desc_off)

    out = {"devices": []}
    for dev in sorted(seen.keys()):
        token, control, desc_off = seen[dev]
        # Query resolver via sysfs
        resolve_path.write_text(str(token))
        line = resolve_path.read_text().strip()
        parts = parse_resolve_line(line)
        next_off = parts.get('next', 'unresolved')
        leaf_off = parts.get('leaf', 'unresolved')
        depth = parts.get('depth', '0')
        out['devices'].append({
            'device_id': f'0x{dev:04x}',
            'token': f'0x{token:08x}',
            'control': f'0x{control:08x}',
            'descriptor_offset': f'0x{desc_off:04x}',
            'next_offset': next_off,
            'leaf_offset': leaf_off,
            'depth': depth,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f'Wrote {OUT} with {len(out["devices"])} devices')


if __name__ == '__main__':
    main()

