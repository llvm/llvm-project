# Check that we don't accidentally strip addr32 prefix

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -nostdlib
# RUN: llvm-objdump -d %t.exe | FileCheck %s
# RUN: llvm-bolt %t.exe -o %t.out -lite=0 -x86-strip-redundant-address-size=false
# RUN: llvm-objdump -d %t.out | FileCheck %s
# CHECK: 67 e8 {{.*}} addr32 callq {{.*}} <foo>
# RUN: llvm-bolt %t.exe -o %t.out -lite=0 -x86-strip-redundant-address-size=true
# remove test name from objdump output, to only search for addr32 in disassembly
# RUN: llvm-objdump -d %t.out | grep -v addr32.s | FileCheck %s --check-prefix=CHECK-STRIP
# CHECK-STRIP-NOT: addr32

.globl	_start
.type	_start, @function
_start:
.code64
  addr32 callq foo
  ret
  .size	_start, .-_start

.globl	foo
.type	foo, @function
foo:
  ud2
