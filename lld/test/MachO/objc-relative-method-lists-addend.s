# REQUIRES: x86
## Symbol referents with a nonzero addend must keep the addend when a method
## list is rewritten with relative offsets.
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11 %t/input.s -o %t/input.o
# RUN: %no-arg-lld -dylib -arch x86_64 -platform_version macos 11 11 \
# RUN:   -objc_relative_method_lists %t/input.o -o %t/out.dylib
# RUN: %python %t/check.py %t/out.dylib

#--- input.s
.text
.globl _imp
_imp:
  retq

.section __TEXT,__objc_methname,cstring_literals
Lsel:
  .asciz "method"
.section __TEXT,__objc_methtype,cstring_literals
.globl _typestr
_typestr:
  .asciz "XXv16@0:8"

.section __DATA,__objc_const
.p2align 3
__OBJC_$_INSTANCE_METHODS_Foo:
  .long 24
  .long 1
  .quad Lsel
  .quad _typestr + 2
  .quad _imp

.section __DATA,__data
.globl _keep
.p2align 3
_keep:
  .quad __OBJC_$_INSTANCE_METHODS_Foo

.subsections_via_symbols

#--- check.py
import struct, sys
data = open(sys.argv[1], "rb").read()
ncmds = struct.unpack_from("<I", data, 16)[0]
pos, sections = 32, {}
for _ in range(ncmds):
    cmd, size = struct.unpack_from("<II", data, pos)
    if cmd == 0x19:
        for i in range(struct.unpack_from("<I", data, pos + 64)[0]):
            sec = pos + 72 + i * 80
            name = data[sec:sec + 16].split(b"\0")[0]
            addr, size, off = struct.unpack_from("<QQI", data, sec + 32)
            sections[name] = (addr, size, off)
    pos += size
addr, _, off = sections[b"__objc_methlist"]
assert struct.unpack_from("<I", data, off)[0] == 0x8000000C
field = addr + 12
target = field + struct.unpack_from("<i", data, off + 12)[0]
taddr, _, toff = next(v for v in sections.values() if v[0] <= target < v[0] + v[1])
s = data[toff + target - taddr:].split(b"\0")[0]
assert s == b"v16@0:8", s
