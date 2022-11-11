// RUN: %clang -### -S -fvisibility=hidden -fvisibility=default %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK-1 %s < %t.log
// CHECK-1-NOT: "-ftype-visibility"
// CHECK-1: "-fvisibility=default"
// CHECK-1-NOT: "-ftype-visibility"

// RUN: %clang -### -S -fvisibility=default -fvisibility=hidden %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK-2 %s < %t.log
// CHECK-2-NOT: "-ftype-visibility"
// CHECK-2: "-fvisibility=hidden"
// CHECK-2-NOT: "-ftype-visibility"

// RUN: %clang -### -S -fvisibility-ms-compat -fvisibility=hidden %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK-3 %s < %t.log
// CHECK-3-NOT: "-ftype-visibility"
// CHECK-3: "-fvisibility=hidden"
// CHECK-3-NOT: "-ftype-visibility"

// RUN: %clang -### -S -fvisibility-ms-compat -fvisibility=default %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK-4 %s < %t.log
// CHECK-4-NOT: "-ftype-visibility"
// CHECK-4: "-fvisibility=default"
// CHECK-4-NOT: "-ftype-visibility"

// RUN: %clang -### -S -fvisibility=hidden -fvisibility-ms-compat %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK-5 %s < %t.log
// CHECK-5: "-fvisibility=hidden"
// CHECK-5: "-ftype-visibility=default"

// RUN: %clang -### -S -fvisibility=default -fvisibility-ms-compat %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK-6 %s < %t.log
// CHECK-6: "-fvisibility=hidden"
// CHECK-6: "-ftype-visibility=default"

