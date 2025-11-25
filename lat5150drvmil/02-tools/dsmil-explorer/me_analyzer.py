#!/usr/bin/env python3
"""
Simple ME dump analyzer for DSMIL explorer.

Checks for common Intel CSE/ME markers, computes a CRC32 over the first
64KB, and reports whether an early ~4.7KB slack (all 0x00) region exists.

Usage:
  python3 02-tools/dsmil-explorer/me_analyzer.py --input 04-hardware/microcode/me_dump.bin
"""
import argparse
import binascii
from pathlib import Path

SIGS = [b"MN2", b"$MN2", b"FTPR", b"$MAN", b"BUP", b"ME"]


def longest_zero_run(b: bytes, limit: int = 64 * 1024) -> tuple[int, int]:
    data = b[:limit]
    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i, v in enumerate(data):
        if v == 0:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_start, best_len


def scan_signatures(b: bytes, limit: int = 2 * 1024 * 1024) -> dict[str, list[int]]:
    data = b[:limit]
    found: dict[str, list[int]] = {}
    for sig in SIGS:
        key = sig.decode('latin1', 'ignore')
        found[key] = []
        start = 0
        while True:
            idx = data.find(sig, start)
            if idx < 0:
                break
            found[key].append(idx)
            start = idx + 1
    return found


def main():
    ap = argparse.ArgumentParser(description="Analyze ME dump for basic structure and slack")
    ap.add_argument('--input', '-i', default='04-hardware/microcode/me_dump.bin', help='Path to ME dump')
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"Input not found: {p}")
        return 1

    b = p.read_bytes()
    crc = binascii.crc32(b[:64 * 1024]) & 0xFFFFFFFF
    start, length = longest_zero_run(b)
    sigs = scan_signatures(b)

    print("ME Dump Analysis")
    print(f"- File: {p}")
    print(f"- Size: {len(b)} bytes")
    print(f"- CRC32(first 64KB): 0x{crc:08x}")

    interesting = {k: v for k, v in sigs.items() if v}
    if interesting:
        print("- Signatures found (first 2MB scan):")
        for k, offs in interesting.items():
            lo = ', '.join(f"+0x{off:x}" for off in offs[:5])
            more = f" (+{len(offs)-5} more)" if len(offs) > 5 else ""
            print(f"  {k}: {lo}{more}")
    else:
        print("- No common CSE/ME markers seen in first 2MB")

    if length >= 4700:
        print(f"- Slack region: YES (offset +0x{start:x}, length ~{length} bytes)")
    elif length >= 4096:
        print(f"- Slack region: POSSIBLE (offset +0x{start:x}, length {length} bytes)")
    else:
        print("- Slack region: not detected (>=4KB run of 0x00)")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

