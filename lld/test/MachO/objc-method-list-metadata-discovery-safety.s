# REQUIRES: aarch64, x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 %t/identity.s -o %t/identity.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --objc-method-list-metadata-discovery --icf=all %t/identity.o -o %t/identity
# RUN: %python %t/check.py identity %t/identity
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 %t/selector.s -o %t/selector.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --objc-method-list-metadata-discovery --icf=all %t/selector.o -o %t/selector
# RUN: %python %t/check.py selector %t/selector
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos13 %t/far.s -o %t/far.o
# RUN: %no-arg-lld -dylib -arch x86_64 -platform_version macos 13 13 -no_fixup_chains -no_objc_relative_method_lists %t/far.o -o %t/far-absolute
# RUN: %python %t/check.py far %t/far-absolute
# RUN: not %no-arg-lld -dylib -arch x86_64 -platform_version macos 13 13 -no_fixup_chains --objc-method-list-metadata-discovery %t/far.o -o %t/far-relative 2>&1 | FileCheck %s --check-prefix=RANGE
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 %t/end.s -o %t/end.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --objc-method-list-metadata-discovery %t/end.o -o %t/end
# RUN: %python %t/check.py end %t/end
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 -defsym=ONLY_NAMED_END=1 %t/end.s -o %t/end-named.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --objc-method-list-metadata-discovery %t/end-named.o -o %t/end-named
# RUN: %python %t/check.py end-named %t/end-named
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 %t/duplicate.s -o %t/duplicate.o
# RUN: %python %t/duplicate.py %t/duplicate.o %t/duplicate-patched.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --objc-method-list-metadata-discovery --enable-objc-protocol-relative-method-lists %t/duplicate-patched.o -o %t/duplicate
# RUN: %python %t/check.py duplicate %t/duplicate

# RANGE: error: {{.*}}ObjC method-list relative offset{{.*}}out of range

#--- identity.s
.text
.globl _get_a, _get_b, _imp
.p2align 2
_get_a:
  adrp x0, _list_a@PAGE
  add x0, x0, _list_a@PAGEOFF
  ret
_get_b:
  adrp x0, _list_b@PAGE
  add x0, x0, _list_b@PAGEOFF
  ret
_imp:
  mov x0, #123
  ret
.section __TEXT,__objc_methname,cstring_literals
Lalpha: .asciz "alpha"
Lbeta: .asciz "beta"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes: .asciz "q16@0:8"
.section __DATA,__objc_const
.p2align 3
.globl _list_a, _list_b
.private_extern _list_a, _list_b
_list_a:
  .long 24, 1
  .quad Lalpha, Ltypes, _imp
_list_b:
  .long 24, 1
  .quad Lbeta, Ltypes, _imp
.section __DATA,__cat_data
.p2align 3
_category_a:
  .quad 0, 0, _list_a, 0, 0, 0, 0
  .long 64, 0
_category_b:
  .quad 0, 0, _list_b, 0, 0, 0, 0
  .long 64, 0
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad _category_a, _category_b
.subsections_via_symbols

#--- selector.s
.text
.globl _imp
.p2align 2
_imp: ret
.section __TEXT,__objc_methname,cstring_literals
.globl _name
.private_extern _name
_name: .asciz "xfoo"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes: .asciz "v16@0:8"
.section __DATA,__objc_const
.p2align 3
.globl _methods
.private_extern _methods
_methods:
  .long 24, 1
  .quad _name, Ltypes, _imp
.section __DATA,__cat_data
.p2align 3
_category:
  .quad 0, 0, _methods, 0, 0, 0, 0
  .long 64, 0
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad _category
.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.globl _suffix_selref
.private_extern _suffix_selref
_suffix_selref: .quad _name + 1
.subsections_via_symbols

#--- far.s
.text
.globl _imp
_imp: retq
.section __TEXT,__objc_methname,cstring_literals
Lsel: .asciz "far"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes: .asciz "v16@0:8"
.section __DATA,__objc_const
.p2align 3
.globl _far_methods
.private_extern _far_methods
_far_methods:
  .long 24, 1
  .quad Lsel, Ltypes, _imp + 0x100000000
.section __DATA,__cat_data
.p2align 3
_category:
  .quad 0, 0, _far_methods, 0, 0, 0, 0
  .long 64, 0
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad _category
.subsections_via_symbols

#--- end.s
.text
.globl _imp
.p2align 2
_imp: ret
.section __TEXT,__objc_methname,cstring_literals
Lsel: .asciz "end"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes: .asciz "v16@0:8"
.section __DATA,__objc_const
.p2align 3
.globl _methods, _list_end
.private_extern _methods, _list_end
_methods:
Lstorage:
  .long 24, 1
  .quad Lsel, Ltypes, _imp
_list_end:
.section __DATA,__cat_data
.p2align 3
_category:
  .quad 0, 0, _methods, 0, 0, 0, 0
  .long 64, 0
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad _category
.section __DATA,__data
.p2align 3
.globl _end_by_name
_end_by_name: .quad _list_end
.ifndef ONLY_NAMED_END
.globl _end_by_section
_end_by_section: .quad Lstorage + 32
.endif
.subsections_via_symbols

#--- duplicate.s
.text
.globl _imp_a, _imp_b
.p2align 2
_imp_a: ret
_imp_b:
  nop
  ret
.section __TEXT,__objc_methname,cstring_literals
Lsel: .asciz "duplicate"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes: .asciz "v16@0:8"
.section __DATA,__objc_const
.p2align 3
.globl _methods
.private_extern _methods
_methods:
  .long 24, 1
  .quad Lsel, Ltypes, _imp_a
.section __DATA,__proto_data
.p2align 3
_protocol:
  .quad 0, 0, 0, _methods, 0, 0, 0, 0
  .long 80, 0
.section __DATA,__objc_protolist,regular,no_dead_strip
.p2align 3
  .quad _protocol
.subsections_via_symbols

#--- duplicate.py
# Append a relocation table containing two UNSIGNED IMP records. This malformed
# input must not be mistaken for a protocol's missing, null IMP relocation.
import struct
import sys
b = bytearray(open(sys.argv[1], "rb").read())
pos = 32
section = None
symbols = {}
for _ in range(struct.unpack_from("<I", b, 16)[0]):
    cmd, size = struct.unpack_from("<II", b, pos)
    if cmd == 0x19:
        for i in range(struct.unpack_from("<I", b, pos + 64)[0]):
            sec = pos + 72 + 80 * i
            if b[sec:sec + 16].split(b"\0")[0] == b"__objc_const":
                section = sec
    elif cmd == 2:
        symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", b, pos + 8)
        for i in range(nsyms):
            idx = struct.unpack_from("<I", b, symoff + 16 * i)[0]
            name = b[stroff + idx:].split(b"\0")[0]
            symbols[bytes(name)] = i
    pos += size
assert section is not None
reloff, nreloc = struct.unpack_from("<II", b, section + 56)
records = bytearray(b[reloff:reloff + nreloc * 8])
for i in range(nreloc):
    address, info = struct.unpack_from("<II", records, i * 8)
    if address == 24:
        assert info & 0x08000000 and info >> 28 == 0
        assert info & 0x00ffffff == symbols[b"_imp_a"]
        second = (info & 0xff000000) | symbols[b"_imp_b"]
        records[i * 8:i * 8] = struct.pack("<II", address, second)
        break
else:
    raise AssertionError("IMP relocation missing")
struct.pack_into("<II", b, section + 56, len(b), nreloc + 1)
b.extend(records)
open(sys.argv[2], "wb").write(b)

#--- check.py
import struct
import sys
blob = open(sys.argv[2], "rb").read()
assert struct.unpack_from("<I", blob)[0] == 0xFEEDFACF
pos, sections, symbols = 32, {}, {}
for _ in range(struct.unpack_from("<I", blob, 16)[0]):
    cmd, size = struct.unpack_from("<II", blob, pos)
    if cmd == 0x19:
        for i in range(struct.unpack_from("<I", blob, pos + 64)[0]):
            sec = pos + 72 + i * 80
            name = blob[sec:sec + 16].split(b"\0")[0]
            sections[name] = struct.unpack_from("<QQI", blob, sec + 32)
    elif cmd == 2:
        symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", blob, pos + 8)
        for i in range(nsyms):
            idx, typ, sect, desc, value = struct.unpack_from("<IBBHQ", blob, symoff + i * 16)
            name = blob[stroff + idx:].split(b"\0")[0]
            symbols[name] = value
    pos += size

def offset(va):
    start, size, off = next(s for s in sections.values() if s[0] <= va < s[0] + s[1])
    return off + va - start

def u64(va):
    return struct.unpack_from("<Q", blob, offset(va))[0]

def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value

def accessor(va):
    adrp, add = struct.unpack_from("<II", blob, offset(va))
    assert adrp & 0x9f000000 == 0x90000000
    assert add & 0xffc00000 == 0x91000000
    pages = signed(((adrp >> 5) & 0x7ffff) << 2 | (adrp >> 29) & 3, 21)
    return (va & ~4095) + (pages << 12) + ((add >> 10) & 4095)

mode = sys.argv[1]
if mode == "identity":
    assert b"__objc_methlist" in sections
    assert symbols[b"_list_a"] != symbols[b"_list_b"]
    assert symbols[b"_get_a"] != symbols[b"_get_b"], "distinct list accessors folded"
    assert accessor(symbols[b"_get_a"]) == symbols[b"_list_a"]
    assert accessor(symbols[b"_get_b"]) == symbols[b"_list_b"]
elif mode == "selector":
    assert b"__objc_methlist" in sections
    field = symbols[b"_methods"] + 8
    ref = field + struct.unpack_from("<i", blob, offset(field))[0]
    actual = blob[offset(u64(ref)):].split(b"\0")[0]
    assert actual == b"xfoo", actual
    suffix = blob[offset(u64(symbols[b"_suffix_selref"])):].split(b"\0")[0]
    assert suffix == b"foo", suffix
elif mode == "far":
    assert b"__objc_methlist" not in sections
    assert u64(symbols[b"_far_methods"] + 24) == symbols[b"_imp"] + 0x100000000
elif mode in ("end", "end-named"):
    assert b"__objc_methlist" not in sections
    expected = symbols[b"_methods"] + 32
    assert symbols[b"_list_end"] == expected
    assert u64(symbols[b"_end_by_name"]) == expected
    if mode == "end":
        assert u64(symbols[b"_end_by_section"]) == expected
elif mode == "duplicate":
    assert b"__objc_methlist" not in sections, "ambiguous IMP was converted"
else:
    raise AssertionError(mode)
