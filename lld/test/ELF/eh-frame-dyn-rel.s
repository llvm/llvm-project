// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: not ld.lld %t.o %t.o -o /dev/null -shared 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK:      error: relocation R_X86_64_64 cannot be used against symbol 'foo'; recompile with -fPIC
// CHECK-NEXT: >>> defined in {{.*}}.o
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.eh_frame+0x12)

.section bar,"axG",@progbits,foo,comdat
.cfi_startproc
.cfi_personality 0x8c, foo
.cfi_endproc
