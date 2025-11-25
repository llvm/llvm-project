#!/usr/bin/env python3
import argparse
from pathlib import Path

SIG = b"$MN2"

def carve_mn2(path, out_dir, context=0x1000):
    data = Path(path).read_bytes()
    offs = []
    i=0
    while True:
        i = data.find(SIG, i)
        if i<0: break
        offs.append(i)
        i += 4
    carved=[]
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    for off in offs:
        start = max(0, off - 256)
        end = min(len(data), off + context)
        chunk = data[start:end]
        p = outp / f"mn2_0x{start:08x}.bin"
        p.write_bytes(chunk)
        carved.append({'offset': start, 'size': len(chunk), 'file': str(p)})
    return offs, carved

def main():
    ap = argparse.ArgumentParser(description='Carve $MN2 regions with context')
    ap.add_argument('path', help='dump path')
    ap.add_argument('--out', default='04-hardware/microcode/ifwi_parts/mn2')
    ap.add_argument('--context', type=lambda x:int(x,0), default=0x2000)
    args = ap.parse_args()
    offs, carved = carve_mn2(args.path, args.out, args.context)
    print(f"MN2 at: {[hex(x) for x in offs]} (carved {len(carved)})")

if __name__ == '__main__':
    main()

