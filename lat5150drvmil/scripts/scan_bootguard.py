#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path

PATTERNS = {
    'ACM': b'ACM!',       # Authenticated Code Module marker (TXT/BootGuard)
    'BPMH': b'BPMH',      # Boot Policy Manifest Header
    'KM': b'KEYM',        # Key Manifest (common text token)
    'SINIT': b'SINIT',    # SINIT ACM hint
    'FIT': b'_FIT_   ',   # FIT directory text
    'FITK': b'FITK',      # Seen in dump strings
}

def scan_all(buf, patterns):
    res = {}
    for name, sig in patterns.items():
        offs = []
        i = 0
        end = len(buf) - len(sig)
        while i <= end:
            if buf[i:i+len(sig)] == sig:
                offs.append(i)
                i += len(sig)
            else:
                i += 1
        res[name] = offs
    return res

def carve_windows(path, hits, out_dir, pre=256, size=32768):
    data = Path(path).read_bytes()
    out = []
    for tag, offs in hits.items():
        for off in offs:
            start = max(0, off - pre)
            end = min(len(data), start + size)
            chunk = data[start:end]
            out_path = Path(out_dir) / f"bg_{tag}_0x{start:08x}.bin"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(chunk)
            out.append({'tag': tag, 'offset': start, 'size': len(chunk), 'file': str(out_path)})
    return out

def main():
    ap = argparse.ArgumentParser(description='Scan firmware dump for BootGuard/ACM/KM/BPM markers and carve windows')
    ap.add_argument('path', help='dump file path')
    ap.add_argument('--out', default='04-hardware/microcode/ifwi_parts/bg', help='output dir')
    ap.add_argument('--json', default='KP14/analysis_results/dumps/ifwi_bg_scan.json', help='json output summary')
    args = ap.parse_args()

    data = Path(args.path).read_bytes()
    hits = scan_all(data, PATTERNS)
    carved = carve_windows(args.path, hits, args.out)
    summary = {
        'file': args.path,
        'size': len(data),
        'hits': {k:[hex(x) for x in v] for k,v in hits.items()},
        'carved': carved,
    }
    Path(Path(args.json).parent).mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {args.json}")
    print(f"Carved {len(carved)} windows to {args.out}")

if __name__ == '__main__':
    main()

