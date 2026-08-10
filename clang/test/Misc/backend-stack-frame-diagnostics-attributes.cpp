// Test the warn-stack-size function attribute is not generated when -Wframe-larger-than is ignored
// through pragma.

// RUN: %clang_cc1 -fwarn-stack-size=70 -emit-llvm -o - %s | FileCheck %s
// CHECK: "warn-stack-size"="70"

// RUN: %clang_cc1 -DIGNORED -fwarn-stack-size=70 -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORED
// IGNORED-NOT: "warn-stack-size"="70"

extern void doIt(char *);

#ifdef IGNORED
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wframe-larger-than"
#endif

void frameSizeAttr() {
  char buffer[80];
  doIt(buffer);
}

#ifdef IGNORED
#pragma GCC diagnostic pop
#endif
