# REQUIRES: aarch64
## On arm64e, a method list is converted only if its IMPs are unsigned or
## signed with the ObjC method-list schema (IA key, address diversity, zero
## discriminator). The runtime signs relative IMPs on use. Lists with any other
## signed field stay absolute, and a signed selref is never reused for a
## relative method name.
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos13 %t/input.s -o %t/input.o
# RUN: %no-arg-lld -dylib -arch arm64e -platform_version macos 13 13 \
# RUN:   -fixup_chains --icf=all --objc-method-list-metadata-discovery \
# RUN:   %t/input.o -o %t/out.dylib
# RUN: %python %t/lists.py %t/out.dylib | FileCheck %s

## Unsigned IMP: converted, and the unsigned input selref is reused.
# CHECK-DAG: mlist_plain __objc_methlist REL n=1 selref_plain v16@0:8 _imp_a
## Standard signed IMP: converted. The only input selref for its selector is
## signed, so a new selref is made.
# CHECK-DAG: mlist_ia_addr __objc_methlist REL n=1 new-selref v16@0:8 _imp_b
## Other signing schemas or signed name/type fields stay absolute.
# CHECK-DAG: mlist_ib __objc_const ABS n=1
# CHECK-DAG: mlist_disc __objc_const ABS n=1
# CHECK-DAG: mlist_noaddr __objc_const ABS n=1
# CHECK-DAG: mlist_signed_sel __objc_const ABS n=1
# CHECK-DAG: mlist_signed_types __objc_const ABS n=1

#--- input.s
.text
.globl _imp_a, _imp_b
.p2align 2
_imp_a:
  ret
_imp_b:
  nop
  ret

.section __TEXT,__objc_methname,cstring_literals
Lsel_plain:
  .asciz "plain"
sel_ia:
  .asciz "ia"
Lsel_ib:
  .asciz "ib"
Lsel_disc:
  .asciz "disc"
Lsel_noaddr:
  .asciz "noaddr"
sel_signed:
  .asciz "signedsel"
Lsel_types:
  .asciz "signedtypes"

.section __TEXT,__objc_methtype,cstring_literals
Ltypes:
  .asciz "v16@0:8"
types_signed:
  .asciz "v24@0:8@16"

.section __TEXT,__cstring,cstring_literals
Lname:
  .asciz "Foo"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
selref_plain:
  .quad Lsel_plain
selref_signed:
  .quad sel_ia@AUTH(da,0)

.section __DATA,__objc_const
.p2align 3
mlist_plain:
  .long 24
  .long 1
  .quad Lsel_plain, Ltypes, _imp_a
mlist_ia_addr:
  .long 24
  .long 1
  .quad sel_ia, Ltypes
  .quad _imp_b@AUTH(ia,0,addr)
mlist_ib:
  .long 24
  .long 1
  .quad Lsel_ib, Ltypes
  .quad _imp_a@AUTH(ib,0,addr)
mlist_disc:
  .long 24
  .long 1
  .quad Lsel_disc, Ltypes
  .quad _imp_a@AUTH(ia,0x1234,addr)
mlist_noaddr:
  .long 24
  .long 1
  .quad Lsel_noaddr, Ltypes
  .quad _imp_a@AUTH(ia,0)
mlist_signed_sel:
  .long 24
  .long 1
  .quad sel_signed@AUTH(da,0)
  .quad Ltypes, _imp_a
mlist_signed_types:
  .long 24
  .long 1
  .quad Lsel_types
  .quad types_signed@AUTH(da,0)
  .quad _imp_a

ro_class:
  .long 0, 8, 8, 0
  .quad 0, Lname, 0, 0, 0, 0, 0
ro_meta:
  .long 1, 40, 40, 0
  .quad 0, Lname, 0, 0, 0, 0, 0

.macro category inst, cls
  .quad Lname, _OBJC_CLASS_$_Foo, \inst, \cls, 0, 0, 0
  .long 64, 0
.endm
cat_a:
  category mlist_plain, mlist_ia_addr
cat_b:
  category mlist_ib, mlist_disc
cat_c:
  category mlist_noaddr, mlist_signed_sel
cat_d:
  category mlist_signed_types, 0

.section __DATA,__objc_data
.p2align 3
.globl _OBJC_CLASS_$_Foo, _OBJC_METACLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .quad _OBJC_METACLASS_$_Foo, 0, 0, 0, ro_class
_OBJC_METACLASS_$_Foo:
  .quad 0, 0, 0, 0, ro_meta

.section __DATA,__objc_classlist,regular,no_dead_strip
.p2align 3
  .quad _OBJC_CLASS_$_Foo
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad cat_a, cat_b, cat_c, cat_d

.subsections_via_symbols

#--- lists.py
## Print each list's section and encoding. Absolute lists hold chained fixups,
## so only relative lists are decoded: the selref's symbol (or new-selref),
## the type string and the IMP symbol.
import struct, sys
data = open(sys.argv[1], "rb").read()
ncmds = struct.unpack_from("<I", data, 16)[0]
pos, sections, symbols = 32, [], {}
for _ in range(ncmds):
    cmd, size = struct.unpack_from("<II", data, pos)
    if cmd == 0x19:
        for i in range(struct.unpack_from("<I", data, pos + 64)[0]):
            sec = pos + 72 + i * 80
            name = data[sec:sec + 16].split(b"\0")[0].decode()
            addr, sz, off = struct.unpack_from("<QQI", data, sec + 32)
            sections.append((name, addr, sz, off))
    elif cmd == 2:
        symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", data, pos + 8)
        for j in range(nsyms):
            idx, typ, sect, desc, value = struct.unpack_from("<IBBHQ", data, symoff + j * 16)
            symbols.setdefault(data[stroff + idx:].split(b"\0")[0].decode(), value)
    pos += size
byaddr = {v: k for k, v in sorted(symbols.items(), reverse=True)}
def sec(va):
    return next(s for s in sections if s[1] <= va < s[1] + s[2])
def u(fmt, va):
    s = sec(va)
    return struct.unpack_from("<" + fmt, data, s[3] + va - s[1])[0]
def cstr(va):
    s = sec(va)
    o = s[3] + va - s[1]
    return data[o:data.index(b"\0", o)].decode()
for name in sorted(symbols):
    if not name.startswith("mlist_"):
        continue
    va = symbols[name]
    s = sec(va)
    flags, count = u("I", va), u("I", va + 4)
    if not flags & 0x80000000:
        print(name, s[0], "ABS", f"n={count}")
        continue
    f = va + 8
    selref = byaddr.get(f + u("i", f), "new-selref")
    types = cstr(f + 4 + u("i", f + 4))
    imp = byaddr.get(f + 8 + u("i", f + 8), "?")
    print(name, s[0], "REL", f"n={count}", selref, types, imp)
