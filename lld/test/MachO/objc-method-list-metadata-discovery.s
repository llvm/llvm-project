# REQUIRES: aarch64
## Method lists are found by following ObjC metadata, not by symbol names. The
## labels below deliberately avoid the compiler's method-list prefixes.
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 %t/input.s -o %t/input.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 \
# RUN:   -no_fixup_chains -undefined dynamic_lookup \
# RUN:   --objc-method-list-metadata-discovery %t/input.o -o %t/meta.dylib
# RUN: %python %t/lists.py %t/meta.dylib | FileCheck %s --check-prefix=META
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 \
# RUN:   -no_fixup_chains -undefined dynamic_lookup -no_objc_relative_method_lists \
# RUN:   --objc-method-list-metadata-discovery %t/input.o -o %t/optout.dylib
# RUN: %python %t/lists.py %t/optout.dylib | FileCheck %s --check-prefix=OPTOUT

## Class and metaclass lists (__objc_classlist).
# META-DAG: mlist_class_inst __objc_methlist REL n=1 alpha v16@0:8 _imp_a
# META-DAG: mlist_class_meta __objc_methlist REL n=1 beta v16@0:8 _imp_b
## Category lists (__objc_catlist).
# META-DAG: mlist_cat_cls __objc_methlist REL n=1 beta v16@0:8 _imp_b
# META-DAG: mlist_cat_inst __objc_methlist REL n=1 alpha v16@0:8 _imp_a
## __objc_catlist2 category; its class field is a Swift class stub, which must
## not be followed as a class_t (otherwise mlist_stub_trap would be converted).
# META-DAG: mlist_cat2 __objc_methlist REL n=1 gamma v16@0:8 _imp_c
## class_ro_t reached directly from __objc_clsrolist (Swift generic classes).
# META-DAG: mlist_clsro __objc_methlist REL n=1 gamma v16@0:8 _imp_c
## No count limit.
# META-DAG: mlist_big __objc_methlist REL n=1100 alpha v16@0:8 _imp_a
## Ineligible lists stay absolute. A bound IMP reads as 0 on disk.
# META-DAG: mlist_bind_imp __objc_const ABS n=1 alpha v16@0:8 NULL
# META-DAG: mlist_interior_ref __objc_const ABS n=2 alpha v16@0:8 _imp_a
# META-DAG: mlist_interior_sym __objc_const ABS n=2 alpha v16@0:8 _imp_a
# META-DAG: mlist_sel_cstring __objc_const ABS n=1 delta v16@0:8 _imp_a
# META-DAG: mlist_stub_trap __objc_const ABS n=1 alpha v16@0:8 _imp_a
## A list embedded in a larger section stays absolute.
# META-DAG: embedded_holder __objc_const
## Already-relative input is left untouched.
# META-DAG: mlist_relative_input __objc_const RELINPUT
## Prefix-named data that is not reachable from metadata is never converted.
# META-DAG: __OBJC_$_INSTANCE_METHODS_Collision __objc_const ABS n=1 alpha v16@0:8 NULL

# OPTOUT-NOT: __objc_methlist

#--- input.s
.text
.globl _imp_a, _imp_b, _imp_c
.p2align 2
_imp_a:
  ret
_imp_b:
  ret
_imp_c:
  ret

.section __TEXT,__objc_methname,cstring_literals
Lsel_a:
  .asciz "alpha"
Lsel_b:
  .asciz "beta"
Lsel_c:
  .asciz "gamma"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes:
  .asciz "v16@0:8"
.section __TEXT,__cstring,cstring_literals
Lname:
  .asciz "Foo"
Lnot_sel:
  .asciz "delta"

.macro entry sel, imp
  .quad \sel
  .quad Ltypes
  .quad \imp
.endm

.section __DATA,__objc_const
.p2align 3
mlist_class_inst:
  .long 24
  .long 1
  entry Lsel_a, _imp_a
mlist_class_meta:
  .long 24
  .long 1
  entry Lsel_b, _imp_b
mlist_cat_inst:
  .long 24
  .long 1
  entry Lsel_a, _imp_a
mlist_cat_cls:
  .long 24
  .long 1
  entry Lsel_b, _imp_b
mlist_big:
  .long 24
  .long 1100
.rept 1100
  entry Lsel_a, _imp_a
.endr
mlist_stub_trap:
  .long 24
  .long 1
  entry Lsel_a, _imp_a
mlist_bind_imp:
  .long 24
  .long 1
  entry Lsel_a, _external_imp
mlist_interior_ref:
  .long 24
  .long 2
  entry Lsel_a, _imp_a
  entry Lsel_b, _imp_b
mlist_interior_sym:
  .long 24
  .long 2
  entry Lsel_a, _imp_a
.alt_entry mlist_interior_sym_second
mlist_interior_sym_second:
  entry Lsel_b, _imp_b
mlist_sel_cstring:
  .long 24
  .long 1
  entry Lnot_sel, _imp_a
embedded_holder:
  .quad 0
  .long 24
  .long 1
  entry Lsel_a, _imp_a
mlist_relative_input:
  .long 0x8000000c
  .long 1
  .long 0
  .long 0
  .long 0
__OBJC_$_INSTANCE_METHODS_Collision:
  .long 24
  .long 1
  entry Lsel_a, 0

ro_class:
  .long 0, 8, 8, 0
  .quad 0, Lname, mlist_class_inst, 0, 0, 0, 0
ro_meta:
  .long 1, 40, 40, 0
  .quad 0, Lname, mlist_class_meta, 0, 0, 0, 0
ro_trap:
  .long 0, 8, 8, 0
  .quad 0, Lname, mlist_stub_trap, 0, 0, 0, 0

.macro category inst, cls, target
  .quad Lname, \target, \inst, \cls, 0, 0, 0
  .long 64, 0
.endm
cat_a:
  category mlist_cat_inst, mlist_cat_cls, _OBJC_CLASS_$_Foo
cat_b:
  category mlist_bind_imp, mlist_interior_ref, _OBJC_CLASS_$_Foo
cat_c:
  category mlist_interior_sym, mlist_sel_cstring, _OBJC_CLASS_$_Foo
cat_d:
  category embedded_holder + 8, mlist_relative_input, _OBJC_CLASS_$_Foo
cat_e:
  category mlist_big, 0, _OBJC_CLASS_$_Foo
cat_stub:
  category mlist_cat2, 0, class_stub

## Swift generic class pattern: class_ro_t lives in __DATA,__const.
.section __DATA,__const
.p2align 3
ro_generic:
  .long 0, 8, 8, 0
  .quad 0, 0, mlist_clsro, 0, 0, 0, 0

.section __DATA,__objc_data
.p2align 3
.globl _OBJC_CLASS_$_Foo, _OBJC_METACLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .quad _OBJC_METACLASS_$_Foo, 0, 0, 0, ro_class
_OBJC_METACLASS_$_Foo:
  .quad 0, 0, 0, 0, ro_meta
## A class stub is {isa = 1, initializer}. Bytes that follow it look like a
## class_t data field pointing at ro_trap.
class_stub:
  .quad 1, _imp_c, 0, 0, ro_trap
mlist_cat2:
  .long 24
  .long 1
  entry Lsel_c, _imp_c
mlist_clsro:
  .long 24
  .long 1
  entry Lsel_c, _imp_c

.section __DATA,__objc_classlist,regular,no_dead_strip
.p2align 3
  .quad _OBJC_CLASS_$_Foo
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad cat_a, cat_b, cat_c, cat_d, cat_e
.section __DATA,__objc_catlist2,regular,no_dead_strip
.p2align 3
  .quad cat_stub
.section __DATA,__objc_clsrolist,regular,no_dead_strip
.p2align 3
  .quad ro_generic

.section __DATA,__data
.globl _keep
.p2align 3
_keep:
  .quad mlist_interior_ref + 32
  .quad __OBJC_$_INSTANCE_METHODS_Collision

.subsections_via_symbols

#--- lists.py
## Print each list symbol's section, encoding and first entry.
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
    if not (name.startswith("mlist_") or name.startswith("__OBJC_$_") or name == "embedded_holder"):
        continue
    if name.endswith("_second"):
        continue
    va = symbols[name]
    s = sec(va)
    flags, count = u("I", va), u("I", va + 4)
    if s[0] != "__objc_methlist" and flags & 0x80000000:
        print(name, s[0], "RELINPUT")
        continue
    if name == "embedded_holder":
        print(name, s[0])
        continue
    f = va + 8
    if flags & 0x80000000:
        sel = cstr(u("Q", f + u("i", f)))
        types = cstr(f + 4 + u("i", f + 4))
        imp = byaddr.get(f + 8 + u("i", f + 8), "?")
        enc = "REL"
    else:
        sel, types = cstr(u("Q", f)), cstr(u("Q", f + 8))
        p = u("Q", f + 16)
        imp = "NULL" if p == 0 else byaddr.get(p, "?")
        enc = "ABS"
    print(name, s[0], enc, f"n={count}", sel, types, imp)
