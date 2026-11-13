# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 -defsym=CACHE=1 %t/input.s -o %t/cache.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --tail-merge-strings --icf=all --objc-method-list-metadata-discovery %t/cache.o -o %t/cache
# RUN: %python %t/check.py cache %t/cache
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 -defsym=DIRECT=1 %t/input.s -o %t/direct.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --tail-merge-strings --icf=all --objc-method-list-metadata-discovery %t/direct.o -o %t/direct
# RUN: %python %t/check.py direct %t/direct
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 -defsym=EXPORT_END=1 %t/input.s -o %t/export.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --tail-merge-strings --icf=all --objc-method-list-metadata-discovery %t/export.o -o %t/export
# RUN: %python %t/check.py export %t/export
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains -dead_strip --tail-merge-strings --icf=all --objc-method-list-metadata-discovery %t/export.o -o %t/export-dead
# RUN: %python %t/check.py export %t/export-dead
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 -defsym=NEGATIVE=1 %t/input.s -o %t/negative.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 -no_fixup_chains --tail-merge-strings --icf=all --objc-method-list-metadata-discovery %t/negative.o -o %t/negative
# RUN: %python %t/check.py negative %t/negative

#--- input.s
.text
.globl _imp
.p2align 2
_imp: ret
.section __TEXT,__objc_methname,cstring_literals
.globl _base, _foo, _inside
.private_extern _base, _foo, _inside
_base:
  .ascii "x"
_inside:
  .asciz "foo"
_foo:
  .asciz "foo"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes: .asciz "v16@0:8"
.section __DATA,__objc_const
.p2align 3
.globl _methods
.ifndef EXPORT_END
.private_extern _methods
.endif
_methods:
  .long 24, 1
.ifdef DIRECT
  .quad _base + 5
.else
.ifdef NEGATIVE
  .quad _inside - 1
.else
  .quad _foo
.endif
.endif
  .quad Ltypes, _imp
.ifdef EXPORT_END
.globl _list_end
_list_end:
.endif
.section __DATA,__cat_data
.p2align 3
_category:
  .quad 0, 0, _methods, 0, 0, 0, 0
  .long 64, 0
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad _category
.ifdef CACHE
.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.globl _existing_selref
.private_extern _existing_selref
_existing_selref:
  .quad _base + 5
.endif
.subsections_via_symbols

#--- check.py
import struct
import sys
b = open(sys.argv[2], "rb").read()
assert struct.unpack_from("<I", b)[0] == 0xFEEDFACF
pos, sections, symbols = 32, {}, {}
for _ in range(struct.unpack_from("<I", b, 16)[0]):
    cmd, size = struct.unpack_from("<II", b, pos)
    if cmd == 0x19:
        for i in range(struct.unpack_from("<I", b, pos + 64)[0]):
            sec = pos + 72 + i * 80
            name = b[sec:sec + 16].split(b"\0")[0]
            sections[name] = struct.unpack_from("<QQI", b, sec + 32)
    elif cmd == 2:
        so, ns, st, sz = struct.unpack_from("<IIII", b, pos + 8)
        for i in range(ns):
            ix, ty, se, de, va = struct.unpack_from("<IBBHQ", b, so + i * 16)
            symbols[b[st + ix:].split(b"\0")[0]] = va
    pos += size

def offset(va):
    a, n, off = next(s for s in sections.values() if s[0] <= va < s[0] + s[1])
    return off + va - a

def u64(va):
    return struct.unpack_from("<Q", b, offset(va))[0]

mode = sys.argv[1]
methods = symbols[b"_methods"]
flags = struct.unpack_from("<I", b, offset(methods))[0]
if mode != "export":
    assert symbols[b"_foo"] == symbols[b"_base"] + 1, "tail merging must occur"
if mode in ("cache", "negative"):
    assert flags == 0x8000000C
    field = methods + 8
    ref = field + struct.unpack_from("<i", b, offset(field))[0]
    target = u64(ref)
    if mode == "cache":
        assert target == symbols[b"_foo"], "cross-piece cache reused"
        assert u64(symbols[b"_existing_selref"]) == symbols[b"_base"] + 5
    else:
        assert target == symbols[b"_base"], "same-piece negative addend changed"
else:
    assert flags == 24 and b"__objc_methlist" not in sections
    if mode == "direct":
        assert u64(methods + 8) == symbols[b"_base"] + 5, "symbol addend reinterpreted"
    elif mode == "export":
        assert symbols[b"_list_end"] == methods + 32, "exported end atom moved"
    else:
        raise AssertionError(mode)
