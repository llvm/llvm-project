#!/usr/bin/env python3
import argparse, os

def scan_holes(path, byte=0xff, min_len=64*1024):
    size = os.path.getsize(path)
    holes = []
    with open(path, 'rb') as f:
        off = 0
        run_b = None
        run_start = 0
        while True:
            chunk = f.read(1024*1024)
            if not chunk:
                break
            i = 0
            n = len(chunk)
            while i < n:
                b = chunk[i]
                if run_b is None:
                    run_b = b
                    run_start = off + i
                    i += 1
                    continue
                if b != run_b:
                    # close run
                    run_end = off + i
                    run_len = run_end - run_start
                    if run_b == byte and run_len >= min_len:
                        holes.append((run_start, run_len))
                    run_b = b
                    run_start = off + i
                i += 1
            off += n
        # finalize
        if run_b is not None:
            run_len = off - run_start
            if run_b == byte and run_len >= min_len:
                holes.append((run_start, run_len))
    return holes

def main():
    ap = argparse.ArgumentParser(description='Find long runs of a single byte (holes) in a binary dump')
    ap.add_argument('path', help='file path')
    ap.add_argument('--byte', type=lambda x:int(x,0), default=0xff, help='byte value to search (default: 0xff)')
    ap.add_argument('--min', dest='min_len', type=lambda x:int(x,0), default=64*1024, help='minimum run length')
    args = ap.parse_args()
    holes = scan_holes(args.path, byte=args.byte, min_len=args.min_len)
    print(f"Found {len(holes)} holes (byte=0x{args.byte:02x}, min={args.min_len})")
    for off,ln in sorted(holes, key=lambda t:t[1], reverse=True)[:20]:
        print(f"  off=0x{off:x} len={ln} (dec {ln})")

if __name__ == '__main__':
    main()

