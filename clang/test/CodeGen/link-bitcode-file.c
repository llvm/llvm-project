// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -mlink-bitcode-file %t.bc -O3 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NO-BC %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE -mlink-bitcode-file %t.bc -O3 -emit-llvm -o - %s 2>&1 | FileCheck -check-prefix=CHECK-BC %s

int f(void);

#ifdef BITCODE

// CHECK-BC: 'f': symbol multiply defined
int f(void) {
  return 42;
}

#else

// CHECK-NO-BC: define i32 @g
// CHECK-NO-BC: ret i32 42
int g(void) {
  return f();
}

// CHECK-NO-BC: define i32 @f

#endif
