#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from cpd_carver import find_cpd, carve

def main():
    ap = argparse.ArgumentParser(description='Scan all $CPD hits and carve entries')
    ap.add_argument('path', help='dump path')
    ap.add_argument('--out', default='04-hardware/microcode/ifwi_parts')
    ap.add_argument('--json', default='KP14/analysis_results/dumps/ifwi_cpd_full.json')
    args = ap.parse_args()
    data = Path(args.path).read_bytes()
    hits = find_cpd(data)
    results = []
    for off in hits:
        rep = carve(args.path, args.out, off)
        results.append({'cpd_off': off, 'entries': rep})
    Path(Path(args.json).parent).mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=2))
    print(f"Found {len(hits)} $CPD hits; wrote {args.json}")

if __name__ == '__main__':
    main()

