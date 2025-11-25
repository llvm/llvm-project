#!/usr/bin/env python3
import argparse
import os
import struct

FIT_SIG = b"_FIT_   "  # 8 bytes (ASCII with spaces)
CPD_SIG = b"$CPD"
MN2_SIG = b"$MN2"


def scan_fit(buf):
    hits = []
    i = 0
    end = len(buf) - len(FIT_SIG)
    while i <= end:
        if buf[i:i+8] == FIT_SIG:
            hits.append(i)
            i += 8
        else:
            i += 1
    return hits

def scan_sig(buf, sig):
    hits = []
    i = 0
    end = len(buf) - len(sig)
    while i <= end:
        if buf[i:i+len(sig)] == sig:
            hits.append(i)
            i += len(sig)
        else:
            i += 1
    return hits


def hexdump(span, base_off=0, width=16):
    out = []
    for i in range(0, len(span), width):
        chunk = span[i:i+width]
        hexs = ' '.join(f"{b:02x}" for b in chunk)
        out.append(f"{base_off+i:08x}  {hexs}")
    return '\n'.join(out)


def try_microcode(buf, off):
    # Heuristic microcode header (48 bytes header)
    try:
        header = buf[off:off+48]
        if len(header) < 48:
            return None
        (hdr_ver, upd_rev, date, sig, checksum, loader_rev,
         plat_id, data_size, total_size) = struct.unpack_from('<9I', header, 0)
        if hdr_ver != 1:
            return None
        # common loader_rev==1; data_size/total_size 1KB multiples
        if data_size % 1024 != 0 or total_size % 1024 != 0:
            return None
        year = (date >> 16) & 0xFFFF
        if year < 2000 or year > 2100:
            return None
        return {
            'hdr_ver': hdr_ver,
            'upd_rev': upd_rev,
            'date': date,
            'sig': sig,
            'checksum': checksum,
            'loader_rev': loader_rev,
            'plat_id': plat_id,
            'data_size': data_size,
            'total_size': total_size,
        }
    except Exception:
        return None


def scan_microcode(buf):
    hits = []
    # scan every 4 bytes
    for off in range(0, len(buf)-48, 4):
        info = try_microcode(buf, off)
        if info:
            hits.append((off, info))
    return hits


def main():
    ap = argparse.ArgumentParser(description='Analyze firmware dump for FIT/ACM/microcode clues')
    ap.add_argument('path', help='Dump file path')
    ap.add_argument('--head', type=int, default=262144, help='Head bytes to scan (default: 256 KiB)')
    ap.add_argument('--tail', type=int, default=262144, help='Tail bytes to scan (default: 256 KiB)')
    args = ap.parse_args()

    size = os.path.getsize(args.path)
    with open(args.path, 'rb') as f:
        head = f.read(min(args.head, size))
        tail = b''
        if size > args.tail:
            f.seek(size - args.tail)
            tail = f.read()

    print(f"File: {args.path} ({size} bytes)")

    # FIT scan
    for label, span, base in (("head", head, 0), ("tail", tail, max(0, size-len(tail)))):
        if not span:
            continue
        hits = scan_fit(span)
        if hits:
            print(f"[+] FIT signature in {label} at offsets: {[hex(base+h) for h in hits]}")
            for h in hits:
                start = max(0, h-64)
                sl = span[start:h+128]
                print(hexdump(sl, base_off=base+start))
                break
        else:
            print(f"[-] No FIT signature in {label}")

    # CPD/MN2 scan (whole file if size is small, otherwise head+tail)
    cpd_hits = []
    mn2_hits = []
    if size <= 32*1024*1024:
        with open(args.path, 'rb') as f:
            data = f.read()
        cpd_hits = scan_sig(data, CPD_SIG)
        mn2_hits = scan_sig(data, MN2_SIG)
    else:
        cpd_hits = [h for h in scan_sig(head, CPD_SIG)]
        mn2_hits = [h for h in scan_sig(head, MN2_SIG)]
        if tail:
            cpd_hits += [size - len(tail) + h for h in scan_sig(tail, CPD_SIG)]
            mn2_hits += [size - len(tail) + h for h in scan_sig(tail, MN2_SIG)]

    if cpd_hits:
        print(f"[+] $CPD signatures at: {[hex(h) for h in cpd_hits[:16]]}{' ...' if len(cpd_hits)>16 else ''}")
        # Show first hit window
        with open(args.path, 'rb') as f:
            f.seek(max(0, cpd_hits[0]-64))
            win = f.read(256)
        print(hexdump(win, base_off=max(0, cpd_hits[0]-64)))
    else:
        print("[-] No $CPD signatures found")

    if mn2_hits:
        print(f"[+] $MN2 signatures at: {[hex(h) for h in mn2_hits[:16]]}{' ...' if len(mn2_hits)>16 else ''}")
        with open(args.path, 'rb') as f:
            f.seek(max(0, mn2_hits[0]-64))
            win = f.read(256)
        print(hexdump(win, base_off=max(0, mn2_hits[0]-64)))
    else:
        print("[-] No $MN2 signatures found")

    # Microcode heuristic scan in head only (fast)
    m_hits = scan_microcode(head)
    if m_hits:
        print(f"[+] Possible microcode headers in head: {len(m_hits)}")
        for off, info in m_hits[:5]:
            print(f"  at +0x{off:x}: {info}")
    else:
        print("[-] No plausible microcode headers detected in head (heuristic)")

    # Dump first 256 bytes of file for table-like patterns
    print("\n[Head 256 bytes preview]")
    print(hexdump(head[:256], base_off=0))

if __name__ == '__main__':
    main()
