#!/usr/bin/env python3
"""Utility for DSMIL driver ioctls (SMI commands, MSR access, ME dump/patch)."""
import argparse
import binascii
import fcntl
import os
import struct
import time
from pathlib import Path

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

MIL_TYPE = ord('M')

SMI_STRUCT = struct.Struct('<HBBIIi')
MSR_STRUCT = struct.Struct('<IIQB7x')
XCR_STRUCT = struct.Struct('<IIQB7x')
PREFERRED_DEVICES = (
    '/dev/dsmil-84dev',
    '/dev/dsmil-72dev',
    '/dev/dsmil',
)


def _IOC(direction, ioc_type, number, size):
    return ((direction << IOC_DIRSHIFT) |
            (ioc_type << IOC_TYPESHIFT) |
            (number << IOC_NRSHIFT) |
            (size << IOC_SIZESHIFT))


def _IOWR(direction, number, size):
    return _IOC(IOC_READ | IOC_WRITE, direction, number, size)

def _IOR(direction, number, size):
    return _IOC(IOC_READ, direction, number, size)

def _IO(direction, number):
    return _IOC(IOC_NONE, direction, number, 0)


def _IOW(direction, number, size):
    return _IOC(IOC_WRITE, direction, number, size)


MILDEV_IOC_SMI_COMMAND = _IOWR(MIL_TYPE, 12, SMI_STRUCT.size)
MILDEV_IOC_MSR_ACCESS = _IOWR(MIL_TYPE, 13, MSR_STRUCT.size)
MILDEV_IOC_XCR_CONTROL = _IOWR(MIL_TYPE, 20, XCR_STRUCT.size)

SMI_EXT_STRUCT = struct.Struct('<HBBHHHBBIIi')
MILDEV_IOC_SMI_COMMAND_EX = _IOWR(MIL_TYPE, 21, SMI_EXT_STRUCT.size)

# ME dump/patch ioctls (as implemented in the driver)
ME_DUMP_REQ = struct.Struct('<QII')
READ_CHUNK = 256
READ_CHUNK_HDR = struct.Struct('<HHIIIQ8x')
MILDEV_IOC_ME_DUMP_START = _IOW(MIL_TYPE, 14, ME_DUMP_REQ.size)
MILDEV_IOC_ME_DUMP_CHUNK = _IOR(MIL_TYPE, 15, READ_CHUNK)
MILDEV_IOC_ME_DUMP_COMPLETE = _IO(MIL_TYPE, 16)
ME_PATCH_REQ = struct.Struct('<IIII')
PATCH_CHUNK_STRUCT = struct.Struct('<HHIIIQ8s224s')
MILDEV_IOC_ME_PATCH_START = _IOW(MIL_TYPE, 17, ME_PATCH_REQ.size)
MILDEV_IOC_ME_PATCH_CHUNK = _IOW(MIL_TYPE, 18, PATCH_CHUNK_STRUCT.size)
MILDEV_IOC_ME_PATCH_COMPLETE = _IO(MIL_TYPE, 19)


def autodetect_device():
    for candidate in PREFERRED_DEVICES:
        if os.path.exists(candidate):
            return candidate
    return PREFERRED_DEVICES[0]


def open_device(path):
    return os.open(path, os.O_RDWR)


def resolve_sysfs_base():
    candidates = [
        Path('/sys/class/dsmil-84dev/dsmil-84dev'),
        Path('/sys/class/dsmil-72dev/dsmil-72dev'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('DSMIL sysfs node not found')


def do_smi(args):
    fd = open_device(args.device)
    try:
        payload = SMI_STRUCT.pack(args.command, args.cpu, args.flags,
                                  args.arg0, args.arg1, 0)
        buf = bytearray(payload)
        try:
            fcntl.ioctl(fd, MILDEV_IOC_SMI_COMMAND, buf, True)
        except OSError as exc:
            print(f"SMI command failed: {exc}")
            return
        _, _, _, _, _, status = SMI_STRUCT.unpack(buf)
        print(f"SMI command 0x{args.command:04x} CPU {args.cpu}: status {status}")
    finally:
        os.close(fd)


def do_msr(args):
    fd = open_device(args.device)
    try:
        write_flag = 1 if args.write else 0
        payload = MSR_STRUCT.pack(args.msr, args.cpu, args.value, write_flag)
        buf = bytearray(payload)
        fcntl.ioctl(fd, MILDEV_IOC_MSR_ACCESS, buf, True)
        msr, cpu, value, _ = MSR_STRUCT.unpack(buf)
        if args.write:
            print(f"Wrote MSR 0x{msr:08x} on CPU {cpu}: 0x{value:016x}")
        else:
            print(f"MSR 0x{msr:08x} on CPU {cpu}: 0x{value:016x}")
    finally:
        os.close(fd)


def do_xcr(args):
    fd = open_device(args.device)
    try:
        write_flag = 1 if args.write is not None else 0
        value = args.write if write_flag else 0
        payload = XCR_STRUCT.pack(args.xcr, args.cpu, value, write_flag)
        buf = bytearray(payload)
        fcntl.ioctl(fd, MILDEV_IOC_XCR_CONTROL, buf, True)
        xcr, cpu, value, _ = XCR_STRUCT.unpack(buf)
        if write_flag:
            print(f"Wrote XCR{xcr} on CPU {cpu}: 0x{value:016x}")
        else:
            print(f"XCR{xcr} on CPU {cpu}: 0x{value:016x}")
    finally:
        os.close(fd)


def do_smi_ext(args):
    fd = open_device(args.device)
    try:
        payload = SMI_EXT_STRUCT.pack(
            args.command, args.cpu, args.flags,
            args.cmd_port, args.data_port, args.trigger_port,
            args.trigger_value, 0, args.arg0, args.arg1, 0
        )
        buf = bytearray(payload)
        try:
            fcntl.ioctl(fd, MILDEV_IOC_SMI_COMMAND_EX, buf, True)
        except OSError as exc:
            print(f"SMI ext failed: {exc}")
            return
        fields = SMI_EXT_STRUCT.unpack(buf)
        status = fields[-1]
        print(f"SMI-EX 0x{args.command:04x} cpu {args.cpu} ports({args.cmd_port:#x},{args.data_port:#x},{args.trigger_port:#x}) → status {status}")
    finally:
        os.close(fd)


def do_smi_probe_extended(args):
    try:
        base = resolve_sysfs_base()
    except FileNotFoundError:
        print('DSMIL sysfs class not found')
        return
    matrix_path = base / 'smi_probe_matrix'
    probe_path = base / 'smi_probe'
    if matrix_path.exists():
        print('[smi_probe_matrix]')
        print(matrix_path.read_text().strip())
    else:
        print('smi_probe_matrix: missing')
    if not probe_path.exists():
        print('smi_probe sysfs attribute unavailable')
        return
    for attempt in range(1, args.count + 1):
        probe_path.write_text('1')
        result = probe_path.read_text().strip()
        print(f'[run {attempt}] {result}')
        if args.delay > 0:
            time.sleep(args.delay)


## (removed duplicate do_me_dump definition)


def do_me_patch(args):
    data = Path(args.input).read_bytes()
    checksum = binascii.crc32(data) & 0xFFFFFFFF
    fd = open_device(args.device)
    try:
        req = ME_PATCH_REQ.pack(len(data), args.slack_offset, checksum, 0)
        fcntl.ioctl(fd, MILDEV_IOC_ME_PATCH_START, req)

        total_chunks = (len(data) + 223) // 224
        offset = 0
        for idx in range(total_chunks):
            chunk_size = min(224, len(data) - offset)
            payload = PATCH_CHUNK_STRUCT.pack(
                0x4d50,  # 'MP'
                idx,
                total_chunks,
                offset,
                chunk_size,
                0,
                b'\x00' * 8,
                data[offset:offset + chunk_size].ljust(224, b'\x00')
            )
            fcntl.ioctl(fd, MILDEV_IOC_ME_PATCH_CHUNK, payload)
            offset += chunk_size

        fcntl.ioctl(fd, MILDEV_IOC_ME_PATCH_COMPLETE, 0)
        print(f"ME patch applied ({len(data)} bytes, CRC 0x{checksum:08x})")
    finally:
        os.close(fd)


def build_parser():
    parser = argparse.ArgumentParser(
        description="DSMIL ioctl helper (SMI and MSR access)")
    parser.add_argument('--device', default=None,
                        help='DSMIL character device (default: auto-detect)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    info = sub.add_parser('info', help='Show driver IOCTL numbers and SMI probe status from sysfs')
    info.set_defaults(func=do_info)

    smi = sub.add_parser('smi', help='Send SMI command via DSMIL driver')
    smi.add_argument('--command', type=lambda x: int(x, 0), required=True,
                     help='SMI command (e.g. 0xA512)')
    smi.add_argument('--cpu', type=lambda x: int(x, 0), default=0,
                     help='Target CPU (or 0xff for all)')
    smi.add_argument('--flags', type=lambda x: int(x, 0), default=0,
                     help='Optional flags byte')
    smi.add_argument('--arg0', type=lambda x: int(x, 0), default=0)
    smi.add_argument('--arg1', type=lambda x: int(x, 0), default=0)
    smi.set_defaults(func=do_smi)

    msr = sub.add_parser('msr', help='Read or write MSRs via DSMIL driver')
    msr.add_argument('--msr', type=lambda x: int(x, 0), required=True,
                     help='MSR address (e.g. 0x1a0)')
    msr.add_argument('--cpu', type=int, default=0,
                     help='CPU index (default: 0)')
    rw = msr.add_mutually_exclusive_group()
    rw.add_argument('--write', type=lambda x: int(x, 0),
                    help='Value to write (hex or decimal)')
    rw.add_argument('--read', action='store_true', help='Force read (default)')
    msr.set_defaults(func=do_msr, value=0, write=None)

    xcr = sub.add_parser('xcr', help='Read or write XCRs via DSMIL driver')
    xcr.add_argument('--xcr', type=lambda x: int(x, 0), default=0,
                     help='XCR index (default: 0)')
    xcr.add_argument('--cpu', type=int, default=0,
                     help='CPU index (default: 0)')
    rwx = xcr.add_mutually_exclusive_group()
    rwx.add_argument('--write', type=lambda x: int(x, 0),
                     help='Value to write (hex or decimal)')
    rwx.add_argument('--read', action='store_true', help='Force read (default)')
    xcr.set_defaults(func=do_xcr)

    smix = sub.add_parser('smi-ex', help='Send extended SMI with custom ports')
    smix.add_argument('--command', type=lambda x: int(x, 0), required=True)
    smix.add_argument('--cpu', type=lambda x: int(x, 0), default=0xff)
    smix.add_argument('--flags', type=lambda x: int(x, 0), default=0)
    smix.add_argument('--cmd-port', type=lambda x: int(x, 0), default=0x164e)
    smix.add_argument('--data-port', type=lambda x: int(x, 0), default=0x164f)
    smix.add_argument('--trigger-port', type=lambda x: int(x, 0), default=0xb2)
    smix.add_argument('--trigger-value', type=lambda x: int(x, 0), default=0xa5)
    smix.add_argument('--arg0', type=lambda x: int(x, 0), default=0)
    smix.add_argument('--arg1', type=lambda x: int(x, 0), default=0)
    smix.set_defaults(func=do_smi_ext)

    probe_ext = sub.add_parser('smi-probe-extended', help='Trigger the read-only SMI probe matrix repeatedly')
    probe_ext.add_argument('--count', type=int, default=1,
                           help='How many times to trigger the probe (default: 1)')
    probe_ext.add_argument('--delay', type=float, default=0.0,
                           help='Delay between probe runs in seconds')
    probe_ext.set_defaults(func=do_smi_probe_extended)

    me = sub.add_parser('me-dump', help='Dump bytes from ME window via DSMIL (chunked)')
    me.add_argument('--offset', type=lambda x: int(x, 0), default=0,
                    help='Offset within ME region (or absolute with --absolute)')
    me.add_argument('--len', dest='length', type=lambda x: int(x, 0), required=True,
                    help='Length to dump (bytes)')
    me.add_argument('--absolute', action='store_true',
                    help='Treat --offset as absolute within DSMIL mapping')
    me.add_argument('--out', required=True, help='Output file path')
    me.set_defaults(func=do_me_dump)

    # (removed duplicate me-dump subparser)

    patch = sub.add_parser('me-patch', help='Stream a custom ME patch into slack region')
    patch.add_argument('--input', '-i', default='04-hardware/microcode/me_patch.bin',
                       help='Source patch binary (default: %(default)s)')
    patch.add_argument('--slack-offset', type=lambda x: int(x, 0), default=0,
                       help='Offset inside current slack region (default: 0)')
    patch.set_defaults(func=do_me_patch)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.device is None:
        args.device = autodetect_device()
    if args.cmd == 'msr':
        if args.write is not None:
            args.value = args.write
            args.write = True
        else:
            args.value = 0
            args.write = False
    args.func(args)


def do_info(args):
    try:
        base = resolve_sysfs_base()
    except FileNotFoundError:
        print('DSMIL sysfs class not found')
        return
    for name in ('ioctl_numbers', 'smi_probe'):
        path = base / name
        if path.exists():
            print(f'[{name}]')
            print(path.read_text().strip())
        else:
            print(f'{name}: not present')


def do_me_dump(args):
    fd = open_device(args.device)
    try:
        flags = 1 if args.absolute else 0
        req = ME_DUMP_REQ.pack(args.offset, args.length, flags)
        # Start session
        fcntl.ioctl(fd, MILDEV_IOC_ME_DUMP_START, req, False)

        # Prepare output file
        written = 0
        with open(args.out, 'wb') as out:
            while written < args.length:
                buf = bytearray(READ_CHUNK)
                fcntl.ioctl(fd, MILDEV_IOC_ME_DUMP_CHUNK, buf, True)
                token, idx, total, data_off, chunk_sz, sess = READ_CHUNK_HDR.unpack(buf[:READ_CHUNK_HDR.size])
                if chunk_sz == 0 or chunk_sz > 224:
                    raise RuntimeError(f'Invalid chunk size {chunk_sz} at index {idx}/{total}')
                data = memoryview(buf)[READ_CHUNK_HDR.size:READ_CHUNK_HDR.size+chunk_sz]
                out.write(data)
                written += chunk_sz
                if idx + 1 >= total:
                    # Completed by count; break to finalize
                    break
        # Complete session
        fcntl.ioctl(fd, MILDEV_IOC_ME_DUMP_COMPLETE, 0)
        print(f'Dumped {written} bytes to {args.out}')
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
