// RUN: %clang -### %s -fescaping-block-tail-calls -fno-escaping-block-tail-calls 2> %t
// RUN: FileCheck --check-prefix=CHECK-DISABLE < %t %s
// CHECK-DISABLE: "-fno-escaping-block-tail-calls"

// RUN: %clang -### %s -fno-escaping-block-tail-calls -fescaping-block-tail-calls 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DISABLE < %t %s
// RUN: %clang -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-DISABLE < %t %s
// CHECK-NO-DISABLE-NOT: "-fno-escaping-block-tail-calls"
