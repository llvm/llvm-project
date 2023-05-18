// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE2 -emit-llvm-bc -o %t-2.bc %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -mlink-bitcode-file %t.bc \
// RUN:     -O3 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NO-BC %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -O3 -emit-llvm -o - \
// RUN:     -mlink-bitcode-file %t.bc -mlink-bitcode-file %t-2.bc %s \
// RUN:     | FileCheck -check-prefix=CHECK-NO-BC -check-prefix=CHECK-NO-BC2 %s
// RUN: not %clang_cc1 -triple i386-pc-linux-gnu -DBITCODE -O3 -emit-llvm -o - \
// RUN:     -mlink-bitcode-file %t.bc %s 2>&1 | FileCheck -check-prefix=CHECK-BC %s
// Make sure we deal with failure to load the file.
// RUN: not %clang_cc1 -triple i386-pc-linux-gnu -mlink-bitcode-file no-such-file.bc \
// RUN:    -emit-llvm -o - %s 2>&1 | FileCheck -check-prefix=CHECK-NO-FILE %s

// Make sure we can perform the same options if the input is LLVM-IR
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm-bc -o %t-in.bc %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -mlink-bitcode-file %t.bc \
// RUN:     -O3 -emit-llvm -o - %t-in.bc | FileCheck -check-prefix=CHECK-NO-BC %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -O3 -emit-llvm -o - \
// RUN:     -mlink-bitcode-file %t.bc -mlink-bitcode-file %t-2.bc %t-in.bc \
// RUN:     | FileCheck -check-prefix=CHECK-NO-BC -check-prefix=CHECK-NO-BC2 %s

int f(void);

#ifdef BITCODE

extern int f2(void);
// CHECK-BC: fatal error: cannot link module {{.*}}'f': symbol multiply defined
int f(void) {
  f2();
  return 42;
}

#elif BITCODE2
int f2(void) { return 43; }
#else

// CHECK-NO-BC-LABEL: define{{.*}} i32 @g
// CHECK-NO-BC: ret i32 42
int g(void) {
  return f();
}

// CHECK-NO-BC-LABEL: define{{.*}} i32 @f
// CHECK-NO-BC2-LABEL: define{{.*}} i32 @f2

#endif

// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file.bc'
