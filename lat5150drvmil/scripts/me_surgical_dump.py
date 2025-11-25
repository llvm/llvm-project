#!/usr/bin/env python3
"""
Surgical ME Dumper using DSMIL ioctls.

Steps:
 1) Scan the first N MB of the DSMIL mapping in 64KB windows using the
    ME_DUMP ioctls in absolute mode to find Intel CSE/ME markers.
 2) Set /sys/class/dsmil-84dev/dsmil-84dev/me_region_override to the
    discovered offset (aligned) and a chosen size (defaults to 8MB).
 3) Perform a relative dump of the entire selected ME region to a file.

Run as root: sudo python3 scripts/me_surgical_dump.py
"""
import argparse
import fcntl
import os
import struct
from pathlib import Path

DEV_CANDIDATES = (
    '/dev/dsmil-84dev',
    '/dev/dsmil-72dev',
    '/dev/dsmil',
)

IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_DIRBITS = 2

IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
IOC_DIRSHIFT = IOC_SIZESHIFT + IOC_SIZEBITS

IOC_NONE = 0
IOC_WRITE = 1
IOC_READ = 2

def _IOC(direction, ioc_type, number, size):
    return ((direction << IOC_DIRSHIFT) |
            (ioc_type << IOC_TYPESHIFT) |
            (number << IOC_NRSHIFT) |
            (size << IOC_SIZESHIFT))

def _IOW(direction, number, size):
    return _IOC(IOC_WRITE, direction, number, size)

def _IOR(direction, number, size):
    return _IOC(IOC_READ, direction, number, size)

def _IO(direction, number):
    return _IOC(IOC_NONE, direction, number, 0)

MIL_TYPE = ord('M')
ME_DUMP_REQ = struct.Struct('<QII')
READ_CHUNK = 256
READ_CHUNK_HDR = struct.Struct('<HHIIIQ8x')
MILDEV_IOC_ME_DUMP_START = _IOW(MIL_TYPE, 14, ME_DUMP_REQ.size)
MILDEV_IOC_ME_DUMP_CHUNK = _IOR(MIL_TYPE, 15, READ_CHUNK)
MILDEV_IOC_ME_DUMP_COMPLETE = _IO(MIL_TYPE, 16)

MARKERS = [b'MN2', b'$MN2', b'FTPR', b'$MAN']


def autodetect_device():
    for c in DEV_CANDIDATES:
        if os.path.exists(c):
            return c
    return DEV_CANDIDATES[0]


def me_dump(fd, offset, length, absolute=False, out_path=None):
    flags = 1 if absolute else 0
    req = ME_DUMP_REQ.pack(offset, length, flags)
    fcntl.ioctl(fd, MILDEV_IOC_ME_DUMP_START, req, False)
    written = 0
    sink = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sink = open(out_path, 'wb')
    try:
        while written < length:
            buf = bytearray(READ_CHUNK)
            fcntl.ioctl(fd, MILDEV_IOC_ME_DUMP_CHUNK, buf, True)
            token, idx, total, data_off, chunk_sz, sess = READ_CHUNK_HDR.unpack(buf[:READ_CHUNK_HDR.size])
            if chunk_sz == 0 or chunk_sz > 224:
                break
            data = memoryview(buf)[READ_CHUNK_HDR.size:READ_CHUNK_HDR.size+chunk_sz]
            if sink:
                sink.write(data)
            written += chunk_sz
            if idx + 1 >= total:
                break
    finally:
        try:
            fcntl.ioctl(fd, MILDEV_IOC_ME_DUMP_COMPLETE, 0)
        except Exception:
            pass
        if sink:
            sink.close()
    return written


def scan_for_me(fd, scan_megs=64, window=64*1024, step=64*1024):
    scan_bytes = scan_megs * 1024 * 1024
    offset = 0
    while offset < scan_bytes:
        # Read a small window in absolute mode (not persisted)
        data_path = None
        length = window
        tmp = bytearray(window)
        # Use the chunked dump to stream into memory: reuse me_dump with no file and local buffer
        # Implement a minimal in-memory: do one shot
        # For efficiency, we can loop READ_CHUNK segments until window consumed, but it's fine to call me_dump per window
        # Here, call me_dump into a temp file path for simplicity avoided; instead dump into /tmp and read back
        tmp_path = f"/tmp/me_scan_{offset:08x}.bin"
        me_dump(fd, offset, length, absolute=True, out_path=tmp_path)
        try:
            with open(tmp_path, 'rb') as f:
                chunk = f.read()
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

        for m in MARKERS:
            if m in chunk:
                return offset + chunk.index(m)
        offset += step
    return None


def write_sysfs(path, text):
    with open(path, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description='Surgical ME dumper')
    ap.add_argument('--device', default=None, help='DSMIL char device (auto)')
    ap.add_argument('--scan-mb', type=int, default=64, help='Scan range in MB (absolute)')
    ap.add_argument('--region-size', type=lambda x: int(x, 0), default=8*1024*1024, help='ME region size to dump')
    ap.add_argument('--out', default='04-hardware/microcode/me_dump.bin', help='Output path')
    args = ap.parse_args()

    dev = args.device or autodetect_device()
    fd = os.open(dev, os.O_RDWR)
    try:
        # 1) Try sysfs me_region first
        me_region_path = '/sys/class/dsmil-84dev/dsmil-84dev/me_region'
        if not os.path.exists(me_region_path):
            me_region_path = '/sys/class/dsmil-72dev/dsmil-72dev/me_region'
        off = size = 0
        valid = 0
        try:
            with open(me_region_path, 'r') as f:
                line = f.read().strip()
                parts = line.split()
                if len(parts) >= 3:
                    valid = int(parts[0])
                    off = int(parts[1], 16)
                    size = int(parts[2], 10)
        except Exception:
            pass

        if not valid:
            # 2) Scan absolute space for markers
            found = scan_for_me(fd, scan_megs=args.scan_mb)
            if found is None:
                raise SystemExit('Could not locate ME markers in scan range')
            region_off = (found // 0x1000) * 0x1000
            region_sz = int(args.region_size)
            # 3) Write overrides
            for p in (
                '/sys/class/dsmil-84dev/dsmil-84dev/me_region_override',
                '/sys/class/dsmil-72dev/dsmil-72dev/me_region_override',
            ):
                if os.path.exists(p):
                    try:
                        write_sysfs(p, f"0x{region_off:x} {region_sz}\n")
                    except Exception:
                        pass
            off, size = region_off, region_sz

        # Ensure ME access gate enabled
        for p in (
            '/sys/class/dsmil-84dev/dsmil-84dev/service_mode_me_access',
            '/sys/class/dsmil-72dev/dsmil-72dev/service_mode_me_access',
        ):
            if os.path.exists(p):
                try:
                    write_sysfs(p, '1\n')
                except Exception:
                    pass

        # 4) Perform relative dump of entire region
        print(f'Starting ME dump: offset=0x{off:x}, size={size} bytes -> {args.out}')
        # Dump in chunks to avoid large kernel allocations
        chunk_size = 1 * 1024 * 1024  # 1MiB per request
        final_path = Path(args.out)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, 'wb') as final:
            total_written = 0
            rel = 0
            while rel < size:
                this_len = min(chunk_size, size - rel)
                tmp_path = f"{args.out}.part_{rel:08x}"
                w = me_dump(fd, rel, this_len, absolute=False, out_path=tmp_path)
                if w <= 0:
                    raise SystemExit(f'Chunk dump failed at +0x{rel:x}')
                with open(tmp_path, 'rb') as part:
                    final.write(part.read())
                os.remove(tmp_path)
                total_written += w
                rel += this_len
        print(f'Dumped {total_written} bytes to {args.out}')
    finally:
        os.close(fd)

if __name__ == '__main__':
    main()
