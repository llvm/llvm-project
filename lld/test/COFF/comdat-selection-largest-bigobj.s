# REQUIRES: x86
#
# Exercise the 32-bit COFF bigobj symbol layout through LARGEST replacement,
# associative-section parent decoding, and secondary-definition tracking.
#
# llvm-mc has no bigobj output switch, so construct one minimal valid bigobj
# directly. Keeping the object small avoids a fixture with > 65279 sections.
#
# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: %python %t.dir/gen-bigobj.py %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
#
# RUN: llvm-objdump -t %t.large.obj | FileCheck %s --check-prefix=BIGOBJ
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.small.obj %t.large.obj %t.root.obj /out:%t.exe
# RUN: llvm-objdump -s %t.exe | FileCheck %s --check-prefix=IMAGE
#
# BIGOBJ: SYMBOL TABLE:
# BIGOBJ: leader
# BIGOBJ: assoc
# BIGOBJ: nested
#
# IMAGE: 44444444
# IMAGE: 55555555
# IMAGE: 66666666
# IMAGE-NOT: 11111111
# IMAGE-NOT: 22222222
# IMAGE-NOT: 33333333
#
#--- small.s
        .section .text$l, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111

        .section .rdata$a, "dr", associative, leader
        .globl assoc
assoc:
        .long 0x22222222

        .section .rdat$b, "dr", associative, assoc
        .globl nested
nested:
        .long 0x33333333
#
#--- root.s
        .text
        .globl entry
entry:
        leaq leader(%rip), %rax
        movl assoc(%rip), %ecx
        movl nested(%rip), %edx
        retq
#
#--- gen-bigobj.py
import struct
import sys

BIGOBJ_MAGIC = bytes.fromhex("c7a1bad1eebaa94baf20faf66aa4dcb8")
MACHINE_AMD64 = 0x8664

SCN_CODE = 0x00000020
SCN_INIT_DATA = 0x00000040
SCN_LNK_COMDAT = 0x00001000
SCN_ALIGN_1 = 0x00100000
SCN_EXEC = 0x20000000
SCN_READ = 0x40000000

STATIC = 3
EXTERNAL = 2
ASSOCIATIVE = 5
LARGEST = 6


def name8(name):
    data = name.encode("ascii")
    assert len(data) <= 8
    return data.ljust(8, b"\0")


def section(name, data, characteristics, raw_ptr):
    return struct.pack(
        "<8sIIIIIIHHI",
        name8(name),
        0,
        0,
        len(data),
        raw_ptr,
        0,
        0,
        0,
        0,
        characteristics,
    )


def symbol(name, value, section_number, storage_class, aux_count=0):
    return struct.pack(
        "<8sIiHBB",
        name8(name),
        value,
        section_number,
        0,
        storage_class,
        aux_count,
    )


def section_aux(length, number, selection):
    # The standard section-definition auxiliary record is 18 bytes. BigObj
    # symbol records are 20 bytes, so append the two bytes of record padding.
    return (
        struct.pack(
            "<IHHIHBBH",
            length,
            0,
            0,
            0,
            number & 0xFFFF,
            selection,
            0,
            number >> 16,
        )
        + b"\0\0"
    )


text = bytes([0x44]) * 36
assoc = bytes.fromhex("55555555")
nested = bytes.fromhex("66666666")
section_data = [text, assoc, nested]

header_size = 56
section_table_size = 3 * 40
first_raw = header_size + section_table_size
raw_ptrs = [
    first_raw,
    first_raw + len(text),
    first_raw + len(text) + len(assoc),
]
symbol_table = first_raw + sum(map(len, section_data))
number_of_symbols = 9

header = struct.pack(
    "<HHHHI16sIIIIIII",
    0,
    0xFFFF,
    2,
    MACHINE_AMD64,
    0,
    BIGOBJ_MAGIC,
    0,
    0,
    0,
    0,
    3,
    symbol_table,
    number_of_symbols,
)

sections = b"".join(
    [
        section(
            ".text$l",
            text,
            SCN_CODE | SCN_LNK_COMDAT | SCN_ALIGN_1 | SCN_EXEC | SCN_READ,
            raw_ptrs[0],
        ),
        section(
            ".rdata$a",
            assoc,
            SCN_INIT_DATA | SCN_LNK_COMDAT | SCN_ALIGN_1 | SCN_READ,
            raw_ptrs[1],
        ),
        section(
            ".rdat$b",
            nested,
            SCN_INIT_DATA | SCN_LNK_COMDAT | SCN_ALIGN_1 | SCN_READ,
            raw_ptrs[2],
        ),
    ]
)

symbols = b"".join(
    [
        symbol(".text$l", 0, 1, STATIC, 1),
        section_aux(len(text), 1, LARGEST),
        symbol("leader", 0, 1, EXTERNAL),
        symbol(".rdata$a", 0, 2, STATIC, 1),
        section_aux(len(assoc), 1, ASSOCIATIVE),
        symbol("assoc", 0, 2, EXTERNAL),
        symbol(".rdat$b", 0, 3, STATIC, 1),
        section_aux(len(nested), 2, ASSOCIATIVE),
        symbol("nested", 0, 3, EXTERNAL),
    ]
)

assert len(header) == header_size
assert len(sections) == section_table_size
assert len(symbols) == number_of_symbols * 20

with open(sys.argv[1], "wb") as out:
    out.write(header)
    out.write(sections)
    out.write(b"".join(section_data))
    out.write(symbols)
    out.write(struct.pack("<I", 4))
