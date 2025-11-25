#!/usr/bin/env python3
import argparse, os, re, struct
from pathlib import Path

SIG_CPD = b"$CPD"

def find_cpd(buf):
    hits = []
    i=0
    while True:
        i = buf.find(SIG_CPD, i)
        if i < 0: break
        hits.append(i)
        i += 4
    return hits

def read_u32le(b, off):
    if off+4>len(b): return None
    return struct.unpack_from('<I', b, off)[0]

def is_printable_name(b):
    try:
        s = b.decode('ascii', 'strict')
    except Exception:
        return False
    # names like 'rbe', 'pgrm', 'kernel', 'sysl', 'fitc.cfg', 'RBEP.man'
    return bool(re.fullmatch(r'[A-Za-z0-9_.-]{3,16}', s))

def carve(file_path, out_dir, cpd_off):
    data = Path(file_path).read_bytes()
    tail = data[cpd_off:]
    # Parse header
    if tail[:4] != SIG_CPD:
        raise SystemExit(f"Not a CPD at 0x{cpd_off:x}")
    # Expected header: $CPD, dword entries, then metadata
    try:
        entry_count = struct.unpack_from('<I', tail, 4)[0]
    except Exception:
        entry_count = 0
    # Heuristic window to scan entries following CPD header (expand for safety)
    scan = tail[:0x20000]
    entries = []
    i = 0x10  # skip CPD header region a bit
    seen = set()
    # Pass 1: name-first pattern
    while i < len(scan)-64 and len(entries) < (entry_count or 1<<30):
        end = scan.find(b'\x00', i, i+32)
        if end>i and end-i<=16 and i>=0x10:
            name_bytes = scan[i:end]
            if is_printable_name(name_bytes):
                name = name_bytes.decode('ascii')
                # look ahead for two dwords (offset,size), allow 0 padding
                probe = end+1
                while probe < end+16 and probe < len(scan)-8 and scan[probe] == 0:
                    probe += 1
                off = read_u32le(scan, probe)
                size = read_u32le(scan, probe+4)
                if off is not None and size is not None and 0 < size <= len(tail) and off < len(tail) and off+size <= len(tail):
                    key=(name,off)
                    if key not in seen:
                        entries.append((name, off, size, cpd_off+off, probe))
                        seen.add(key)
                        i = probe + 8
                        continue
        i += 1

    # Pass 2: off/size-first pattern (fitc.cfg etc.)
    i = 0x10
    while i < len(scan)-64 and len(entries) < (entry_count or 1<<30):
        off = read_u32le(scan, i)
        size = read_u32le(scan, i+4)
        if off is not None and size is not None and 0 < size <= len(tail) and off < len(tail) and off+size <= len(tail):
            # skip 8 bytes of zeros/reserved if present
            probe = i+8
            zero_pad = 0
            while zero_pad < 8 and probe < len(scan) and scan[probe] == 0:
                probe += 1; zero_pad += 1
            # attempt to read a name
            end = scan.find(b'\x00', probe, probe+32)
            if end>probe and end-probe<=16:
                nb = scan[probe:end]
                if is_printable_name(nb):
                    name = nb.decode('ascii')
                    key=(name,off)
                    if key not in seen:
                        entries.append((name, off, size, cpd_off+off, i))
                        seen.add(key)
                        i = end + 1
                        continue
        i += 1

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    report = []
    for name, off, size, abs_off, loc in entries:
        chunk = tail[off:off+size]
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
        out_file = outp / f"{safe}_0x{abs_off:08x}_0x{size:x}.bin"
        with open(out_file, 'wb') as f:
            f.write(chunk)
        report.append((name, abs_off, size, str(out_file)))

    return report

def main():
    ap = argparse.ArgumentParser(description='Heuristic CPD carver')
    ap.add_argument('path', help='dump binary')
    ap.add_argument('--cpd-off', type=lambda x:int(x,0), default=None, help='CPD offset (hex or dec); if omitted, use last $CPD in file')
    ap.add_argument('--out', default='04-hardware/microcode/ifwi_parts', help='output dir')
    args = ap.parse_args()
    data = Path(args.path).read_bytes()
    off = args.cpd_off
    if off is None:
        hits = find_cpd(data)
        if not hits:
            print('No $CPD found')
            return 1
        off = hits[-1]
    rep = carve(args.path, args.out, off)
    print(f"Carved {len(rep)} entries from $CPD at 0x{off:x}")
    for name, abs_off, size, path in rep:
        print(f"  {name:12s} off=0x{abs_off:08x} size=0x{size:x} -> {path}")

if __name__ == '__main__':
    main()
