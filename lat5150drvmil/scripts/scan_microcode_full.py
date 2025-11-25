#!/usr/bin/env python3
import argparse, struct, json
from pathlib import Path

def parse_ucode_header(buf, off):
    # Intel microcode header is 48 bytes
    if off+48 > len(buf):
        return None
    try:
        hdr_ver, uprid, date, sig, checksum, loader_rev, plat_id, data_size, total_size = struct.unpack_from('<9I', buf, off)
    except struct.error:
        return None
    # Heuristics
    if hdr_ver != 1:
        return None
    if data_size % 1024 != 0 or total_size % 1024 != 0 or total_size == 0:
        return None
    year = (date >> 16) & 0xFFFF
    if not (2000 <= year <= 2100):
        return None
    return {
        'offset': off,
        'hdr_ver': hdr_ver,
        'upd_rev': uprid,
        'date': date,
        'cpu_sig': sig,
        'checksum': checksum,
        'loader_rev': loader_rev,
        'platform_id': plat_id,
        'data_size': data_size,
        'total_size': total_size,
    }

def scan_file(path, step=4):
    data = Path(path).read_bytes()
    hits = []
    for off in range(0, len(data)-48, step):
        info = parse_ucode_header(data, off)
        if info:
            hits.append(info)
    return hits

def main():
    ap = argparse.ArgumentParser(description='Scan binary for Intel microcode headers')
    ap.add_argument('path', help='binary dump')
    ap.add_argument('--json', help='write JSON summary')
    args = ap.parse_args()
    hits = scan_file(args.path)
    for h in hits:
        y=(h['date']>>16)&0xFFFF; m=(h['date']>>8)&0xFF; d=h['date']&0xFF
        print(f"0x{h['offset']:08x} sig=0x{h['cpu_sig']:08x} plat=0x{h['platform_id']:08x} date={y:04d}-{m:02d}-{d:02d} total=0x{h['total_size']:x}")
    if args.json:
        Path(args.json).write_text(json.dumps(hits, indent=2))
        print(f"Wrote {args.json}")

if __name__ == '__main__':
    main()

