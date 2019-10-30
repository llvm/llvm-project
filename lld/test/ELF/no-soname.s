// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: rm -rf %t.dir && mkdir -p %t.dir/no-soname
// RUN: ld.lld %t.o -shared -o %t.dir/no-soname/libfoo.so

// RUN: ld.lld %t.o %t.dir/no-soname/libfoo.so -o %t
// RUN: llvm-readobj --dynamic-table %t | FileCheck %s

// CHECK:  0x0000000000000001 NEEDED               Shared library: [{{.*}}/no-soname/libfoo.so]
// CHECK-NOT: NEEDED

// RUN: ld.lld %t.o %t.dir/no-soname/../no-soname/libfoo.so -o %t
// RUN: llvm-readobj --dynamic-table %t | FileCheck %s --check-prefix=CHECK2

// CHECK2:  0x0000000000000001 NEEDED               Shared library: [{{.*}}/no-soname/../no-soname/libfoo.so]
// CHECK2-NOT: NEEDED

// RUN: ld.lld %t.o -L%t.dir/no-soname/../no-soname -lfoo -o %t
// RUN: llvm-readobj --dynamic-table %t | FileCheck %s --check-prefix=CHECK3

// CHECK3:  0x0000000000000001 NEEDED               Shared library: [libfoo.so]
// CHECK3-NOT: NEEDED

// RUN: ld.lld %t.o -shared -soname libbar.so -o %t.dir/no-soname/libbar.so
// RUN: ld.lld %t.o %t.dir/no-soname/libbar.so -o %t
// RUN: llvm-readobj --dynamic-table %t | FileCheck %s --check-prefix=CHECK4

// CHECK4:  0x0000000000000001 NEEDED               Shared library: [libbar.so]
// CHECK4-NOT: NEEDED

.global _start
_start:
