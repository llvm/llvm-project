// RUN: %clang --target=x86_64-unknown-ananas -static %s \
// RUN:   --sysroot=%S/Inputs/ananas-tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC: ld{{.*}}" "-Bstatic"
// CHECK-STATIC: crt0.o
// CHECK-STATIC: crti.o
// CHECK-STATIC: crtbegin.o
// CHECK-STATIC: crtend.o
// CHECK-STATIC: crtn.o

// RUN: %clang --target=x86_64-unknown-ananas -shared %s \
// RUN:   --sysroot=%S/Inputs/ananas-tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED: crti.o
// CHECK-SHARED: crtbeginS.o
// CHECK-SHARED: crtendS.o
// CHECK-SHARED: crtn.o

// -r suppresses -dynamic-linker, default -l and crt*.o like -nostdlib.
// RUN: %clang %s -### -o %t.o --target=x86_64-unknown-ananas -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE:     "-r"
// CHECK-RELOCATABLE-NOT: "-pie"
// CHECK-RELOCATABLE-NOT: "-dynamic-linker"
// CHECK-RELOCATABLE-NOT: "-l
// CHECK-RELOCATABLE-NOT: /crt{{[^.]+}}.o
