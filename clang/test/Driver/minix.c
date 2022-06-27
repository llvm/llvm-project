// -r suppresses default -l and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=i386-unknown-minix -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE:     "-r"
// CHECK-RELOCATABLE-NOT: "-l
// CHECK-RELOCATABLE-NOT: /crt{{[^.]+}}.o
