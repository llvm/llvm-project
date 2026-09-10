#!/usr/bin/env python3
"""Minimal zebin (Intel GPU native binary) writer/reader for M0.

Ground truth: NEO shared/source/device_binary_format/zebin/.
Emits an ET_ZEBIN_EXE ELF64 with .text.<kernel>, .symtab, .ze_info and
.note.intelgt.compat sections. No relocations, no SPIR-V.
"""

import argparse
import struct
import sys

SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_NOTE = 7
SHT_ZEBIN_ZEINFO = 0xFF000011
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
ET_ZEBIN_EXE = 0xFF12
EM_INTEL_GT = 0xCD


def align(x, a):
    return (x + a - 1) // a * a


def elf_sections(data):
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    _, _, soff, ssz, _ = _shdr(data, shoff, shentsize, shstrndx)
    shstr = data[soff:soff + ssz]
    out = {}
    for i in range(shnum):
        name_off, typ, off, size, link = _shdr(data, shoff, shentsize, i)
        end = shstr.find(b"\0", name_off)
        out[shstr[name_off:end].decode()] = (typ, off, size, link)
    return out


def _shdr(data, shoff, shentsize, i):
    base = shoff + i * shentsize
    name, typ = struct.unpack_from("<II", data, base)
    off, size = struct.unpack_from("<QQ", data, base + 0x18)
    link = struct.unpack_from("<I", data, base + 0x28)[0]
    return name, typ, off, size, link


def cmd_extract(args):
    data = open(args.input, "rb").read()
    secs = elf_sections(data)
    for name, (_, off, size, _) in secs.items():
        if name.startswith(".text."):
            kname = name[len(".text."):]
            open(f"{args.outdir}/{kname}.text.bin", "wb").write(data[off:off + size])
            print(f"kernel {kname}: {size} bytes")
    for want, fname in ((".ze_info", "zeinfo.yaml"),
                        (".note.intelgt.compat", "note.compat.bin")):
        if want not in secs:
            raise SystemExit(f"missing required section {want}")
        _, off, size, _ = secs[want]
        open(f"{args.outdir}/{fname}", "wb").write(data[off:off + size])
        print(f"{want}: {size} bytes -> {fname}")


def cmd_write(args):
    text = open(args.text, "rb").read()
    zeinfo = open(args.zeinfo, "rb").read()
    notes = open(args.notes, "rb").read()
    kname = args.kernel.encode()

    # (name, type, flags, align, data)
    sections = [
        (f".text.{args.kernel}".encode(), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 64, text),
        (b".symtab", SHT_SYMTAB, 0, 8, None),  # filled below
        (b".strtab", SHT_STRTAB, 0, 1, None),
        (b".ze_info", SHT_ZEBIN_ZEINFO, 0, 8, zeinfo),
        (b".note.intelgt.compat", SHT_NOTE, 0, 4, notes),
    ]

    strtab = b"\0" + kname + b"\0"
    kname_off = 1
    # null sym + STB_GLOBAL|STT_FUNC symbol for the kernel in .text.
    symtab = b"\0" * 24
    symtab += struct.pack("<IBBHQQ", kname_off, 0x12, 0, 1, 0, len(text))
    sections[1] = (sections[1][0], SHT_SYMTAB, 0, 8, symtab)
    sections[2] = (sections[2][0], SHT_STRTAB, 0, 1, strtab)

    shstr = b"\0"
    name_offs = []
    for name, *_ in sections:
        name_offs.append(len(shstr))
        shstr += name + b"\0"
    shstr_name_off = len(shstr)
    shstr += b".shstrtab\0"

    ehdr_size = 64
    off = ehdr_size
    placed = []
    for name, typ, flags, al, data in sections:
        off = align(off, max(al, 1))
        placed.append((name, typ, flags, al, off, data))
        off += len(data)
    shstr_off = off
    off += len(shstr)
    shoff = align(off, 8)

    etype = args.etype
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    ehdr = ident + struct.pack(
        "<HHIQQQIHHHHHH",
        etype, EM_INTEL_GT, 1, 0, 0, shoff, 0, 64, 0, 0, 64,
        len(sections) + 2, len(sections) + 1)

    blob = bytearray()
    blob += ehdr
    for name, typ, flags, al, soff, data in placed:
        blob += b"\0" * (soff - len(blob))
        blob += data
    blob += b"\0" * (shstr_off - len(blob))
    blob += shstr
    blob += b"\0" * (shoff - len(blob))

    strtab_idx = next(i for i, p in enumerate(placed, 1) if p[0] == b".strtab")
    blob += b"\0" * 64  # section header 0 is always NULL.
    for i, (name, typ, flags, al, soff, data) in enumerate(placed, 1):
        link = strtab_idx if typ == SHT_SYMTAB else 0
        info = 1 if typ == SHT_SYMTAB else 0  # one local symbol
        entsize = 24 if typ == SHT_SYMTAB else 0
        blob += struct.pack("<IIQQQQIIQQ", name_offs[i - 1], typ, flags, 0,
                            soff, len(data), link, info, al, entsize)
    blob += struct.pack("<IIQQQQIIQQ", shstr_name_off, SHT_STRTAB, 0, 0,
                        shstr_off, len(shstr), 0, 0, 1, 0)

    open(args.output, "wb").write(blob)
    print(f"wrote {args.output}: {len(blob)} bytes, e_type={hex(etype)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="extract sections from a zebin")
    ex.add_argument("input")
    ex.add_argument("outdir")
    ex.set_defaults(fn=cmd_extract)
    wr = sub.add_parser("write", help="build a zebin EXE")
    wr.add_argument("--kernel", required=True)
    wr.add_argument("--text", required=True)
    wr.add_argument("--zeinfo", required=True)
    wr.add_argument("--notes", required=True)
    wr.add_argument("--etype", type=lambda s: int(s, 0), default=ET_ZEBIN_EXE)
    wr.add_argument("-o", "--output", required=True)
    wr.set_defaults(fn=cmd_write)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
