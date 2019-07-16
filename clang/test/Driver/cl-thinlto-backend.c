// RUN: %clang_cl -c -flto=thin -Fo%t.obj -- %s
// RUN: llvm-lto2 run -thinlto-distributed-indexes -o %t.exe %t.obj

// -fthinlto_index should be passed to cc1
// RUN: %clang_cl -### -c -fthinlto-index=%t.thinlto.bc -Fo%t1.obj \
// RUN:     -- %t.obj 2>&1 | FileCheck %s

// CHECK: -fthinlto-index=
// CHECK: "-x" "ir"
