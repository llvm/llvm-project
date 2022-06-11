# This reproduces a bug where not registering cold fragment entry points 
# leads to removing blocks and an inconsistent CFG after UCE.
# Test assembly was obtained using C-Reduce from this C++ code:
# (compiled with `g++ -O2 -Wl,-q`)
#
# #include <stdexcept>
# int a;
# int main() {
#  if (a)
#    try {
#      throw std::logic_error("");
#    } catch (...) {}
#  try {
#    throw std::logic_error("");
#  } catch (...) {}
# }

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clangxx %cxxflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --dump-dot-all --funcs=main.* \
# RUN:   2>&1 | FileCheck %s
#
# CHECK-NOT: Assertion `isValid()' failed.

	.globl main
main:
	jmp .L3 # jump to the secondary entry point in main.cold.0

	.section a,"ax"
main.cold.0:
	.cfi_startproc
	.cfi_lsda 3,b
  ud2
  call __cxa_throw
.L3:
  nop
	.cfi_endproc

	.section  .gcc_except_table
b:
  .byte 0xff,0x3
  .uleb128 e-c
c:
  .byte 1
  .uleb128 e-d
d:
  .uleb128 0,0,0,0,0
  .uleb128 .L3-main.cold.0
  .uleb128 .L3-main.cold.0
e:
