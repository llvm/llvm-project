// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fmemory-profile \
// RUN:   -emit-obj %t/a.cpp -o %t/a.o
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fmemory-profile \
// RUN:   -emit-obj %t/b.cpp -o %t/b.o
// RUN: llvm-readelf --notes %t/a.o | FileCheck %s --check-prefix=NOTE
// RUN: llvm-readelf --notes %t/b.o | FileCheck %s --check-prefix=NOTE
// RUN: llvm-objcopy --dump-section \
// RUN:   .note.llvm.memprof.tu-id=%t/a.note %t/a.o
// RUN: llvm-objcopy --dump-section \
// RUN:   .note.llvm.memprof.tu-id=%t/b.note %t/b.o
// RUN: not cmp %t/a.note %t/b.note
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj \
// RUN:   %t/a.cpp -o %t/no-memprof.o
// RUN: llvm-readelf --sections %t/no-memprof.o | \
// RUN:   FileCheck %s --check-prefix=NO-NOTE

// NOTE: Displaying notes found in: .note.llvm.memprof.tu-id
// NOTE: LLVM
// NOTE-SAME: 0x00000010
// NO-NOTE-NOT: .note.llvm.memprof.tu-id

//--- a.cpp
int a() { return 1; }

//--- b.cpp
int b() { return 2; }
