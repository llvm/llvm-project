# REQUIRES: aarch64
## Protocol method lists are opt-in. They are reached from __objc_protolist,
## __objc_protorefs, class_ro_t/category protocol lists and inherited protocol
## lists. Their null IMPs are encoded as a zero relative offset.
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos13 %t/input.s -o %t/input.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 \
# RUN:   -no_fixup_chains --objc-method-list-metadata-discovery \
# RUN:   %t/input.o -o %t/default.dylib
# RUN: %python %t/lists.py %t/default.dylib | FileCheck %s --check-prefix=DEFAULT
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 \
# RUN:   -no_fixup_chains --objc-method-list-metadata-discovery \
# RUN:   --enable-objc-protocol-relative-method-lists %t/input.o -o %t/proto.dylib
# RUN: %python %t/lists.py %t/proto.dylib | FileCheck %s --check-prefix=PROTO
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 \
# RUN:   -no_fixup_chains --objc-method-list-metadata-discovery \
# RUN:   --enable-objc-protocol-relative-method-lists -no_objc_relative_method_lists \
# RUN:   %t/input.o -o %t/optout.dylib
# RUN: %python %t/lists.py %t/optout.dylib | FileCheck %s --check-prefix=OPTOUT
# RUN: not %no-arg-lld -dylib -arch arm64 -platform_version macos 13 13 \
# RUN:   --enable-objc-protocol-relative-method-lists %t/input.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR

# DEFAULT-DAG: mlist_class __objc_methlist REL n=1 alpha v16@0:8 _imp_a
# DEFAULT-DAG: mlist_p_req_inst __objc_const ABS n=1 alpha v16@0:8 NULL
# DEFAULT-DAG: mlist_parent_opt_cls __objc_const ABS n=1 delta v16@0:8 NULL

# PROTO-DAG: mlist_class __objc_methlist REL n=1 alpha v16@0:8 _imp_a
## Protocol in __objc_protolist, all four kinds.
# PROTO-DAG: mlist_p_req_inst __objc_methlist REL n=1 alpha v16@0:8 NULL
# PROTO-DAG: mlist_p_req_cls __objc_methlist REL n=1 beta v16@0:8 NULL
# PROTO-DAG: mlist_p_opt_inst __objc_methlist REL n=1 gamma v16@0:8 NULL
# PROTO-DAG: mlist_p_opt_cls __objc_methlist REL n=1 delta v16@0:8 NULL
## Inherited protocol, reached only through P's protocol list.
# PROTO-DAG: mlist_parent_opt_cls __objc_methlist REL n=1 delta v16@0:8 NULL
## Reached only through a class_ro_t protocol list / a category protocol list /
## __objc_protorefs.
# PROTO-DAG: mlist_by_class __objc_methlist REL n=1 alpha v16@0:8 NULL
# PROTO-DAG: mlist_by_category __objc_methlist REL n=1 beta v16@0:8 NULL
# PROTO-DAG: mlist_by_ref __objc_methlist REL n=1 gamma v16@0:8 NULL
## A protocol list that has a nonnull IMP is still a valid method list.
# PROTO-DAG: mlist_imp_in_proto __objc_methlist REL n=1 alpha v16@0:8 _imp_a

# OPTOUT-NOT: __objc_methlist

# ERR: error: --enable-objc-protocol-relative-method-lists requires --objc-method-list-metadata-discovery

#--- input.s
.text
.globl _imp_a
.p2align 2
_imp_a:
  ret

.section __TEXT,__objc_methname,cstring_literals
Lsel_a:
  .asciz "alpha"
Lsel_b:
  .asciz "beta"
Lsel_c:
  .asciz "gamma"
Lsel_d:
  .asciz "delta"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes:
  .asciz "v16@0:8"
.section __TEXT,__cstring,cstring_literals
Lname:
  .asciz "P"

.macro list sel, imp
  .long 24
  .long 1
  .quad \sel, Ltypes, \imp
.endm
.macro protocol inherited, ri, rc, oi, oc
  .quad 0, Lname, \inherited, \ri, \rc, \oi, \oc, 0
  .long 80, 0
.endm

.section __DATA,__objc_const
.p2align 3
mlist_class:
  list Lsel_a, _imp_a
mlist_p_req_inst:
  list Lsel_a, 0
mlist_p_req_cls:
  list Lsel_b, 0
mlist_p_opt_inst:
  list Lsel_c, 0
mlist_p_opt_cls:
  list Lsel_d, 0
mlist_parent_opt_cls:
  list Lsel_d, 0
mlist_by_class:
  list Lsel_a, 0
mlist_by_category:
  list Lsel_b, 0
mlist_by_ref:
  list Lsel_c, 0
mlist_imp_in_proto:
  list Lsel_a, _imp_a
p_inherits:
  .quad 1, proto_parent
class_protocols:
  .quad 1, proto_by_class
category_protocols:
  .quad 1, proto_by_category
ro_class:
  .long 0, 8, 8, 0
  .quad 0, Lname, mlist_class, class_protocols, 0, 0, 0
ro_meta:
  .long 1, 40, 40, 0
  .quad 0, Lname, 0, 0, 0, 0, 0
cat:
  .quad Lname, _OBJC_CLASS_$_Foo, 0, 0, category_protocols, 0, 0
  .long 64, 0

.section __DATA,__data
.p2align 3
proto_p:
  protocol p_inherits, mlist_p_req_inst, mlist_p_req_cls, mlist_p_opt_inst, mlist_p_opt_cls
proto_parent:
  protocol 0, 0, 0, 0, mlist_parent_opt_cls
proto_by_class:
  protocol 0, mlist_by_class, 0, 0, 0
proto_by_category:
  protocol 0, 0, mlist_by_category, 0, 0
proto_by_ref:
  protocol 0, 0, 0, mlist_by_ref, 0
proto_imp:
  protocol 0, mlist_imp_in_proto, 0, 0, 0

.section __DATA,__objc_data
.p2align 3
.globl _OBJC_CLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .quad _OBJC_METACLASS_$_Foo, 0, 0, 0, ro_class
_OBJC_METACLASS_$_Foo:
  .quad 0, 0, 0, 0, ro_meta

.section __DATA,__objc_classlist,regular,no_dead_strip
.p2align 3
  .quad _OBJC_CLASS_$_Foo
.section __DATA,__objc_catlist,regular,no_dead_strip
.p2align 3
  .quad cat
.section __DATA,__objc_protolist,coalesced,no_dead_strip
.p2align 3
  .quad proto_p, proto_imp
.section __DATA,__objc_protorefs,coalesced,no_dead_strip
.p2align 3
  .quad proto_by_ref

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
        delta = u("i", f + 8)
        imp = "NULL" if delta == 0 else byaddr.get(f + 8 + delta, "?")
        enc = "REL"
    else:
        sel, types = cstr(u("Q", f)), cstr(u("Q", f + 8))
        p = u("Q", f + 16)
        imp = "NULL" if p == 0 else byaddr.get(p, "?")
        enc = "ABS"
    print(name, s[0], enc, f"n={count}", sel, types, imp)
