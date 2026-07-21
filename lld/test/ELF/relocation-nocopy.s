// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t.so
// RUN: not ld.lld -z nocopyreloc %t.o %t.so -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK:      error: unresolvable relocation R_X86_64_32S against symbol 'x'; recompile with -fPIC or remove '-z nocopyreloc'
// CHECK-NEXT: >>> defined in {{.*}}.so
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x{{.*}})
// CHECK:      error: unresolvable relocation R_X86_64_32S against symbol 'y'; recompile with -fPIC or remove '-z nocopyreloc'
// CHECK-NEXT: >>> defined in {{.*}}.so
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x{{.*}})
// CHECK:      error: unresolvable relocation R_X86_64_32S against symbol 'z'; recompile with -fPIC or remove '-z nocopyreloc'
// CHECK-NEXT: >>> defined in {{.*}}.so
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x{{.*}})
// CHECK:      error: unresolvable relocation R_X86_64_32 against symbol 'x'; recompile with -fPIC or remove '-z nocopyreloc'
// CHECK-NEXT: >>> defined in {{.*}}.so
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x{{.*}})
// CHECK:      error: unresolvable relocation R_X86_64_32 against symbol 'y'; recompile with -fPIC or remove '-z nocopyreloc'
// CHECK-NEXT: >>> defined in {{.*}}.so
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x{{.*}})
// CHECK:      error: unresolvable relocation R_X86_64_32 against symbol 'z'; recompile with -fPIC or remove '-z nocopyreloc'
// CHECK-NEXT: >>> defined in {{.*}}.so
// CHECK-NEXT: >>> referenced by {{.*}}.o:(.text+0x{{.*}})

.text
.global _start
_start:
movl $5, x
movl $7, y
movl $9, z
movl $x, %edx
movl $y, %edx
movl $z, %edx
