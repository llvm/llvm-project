// -fkeep-inline-functions has an effect without optimization, although it is
// primarily useful with optimization enabled.

// RUN: %clang_cc1 -O2 -fkeep-inline-functions -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -O0 -fkeep-inline-functions -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -O2 -fkeep-inline-functions -emit-llvm %s -o - -triple powerpc64-ibm-aix-xcoff | FileCheck %s
// RUN: %clang_cc1 -O2 -fkeep-inline-functions -fgnu89-inline -emit-llvm %s -o - -triple powerpc64-ibm-aix-xcoff | FileCheck %s --check-prefix=CHECK-GNU89

// Retained:
//   f1  static inline
//   f2  plain inline under -fgnu89-inline
//   f5  C99 external definition with external linkage;
//       considered plain inline under -fgnu89-inline
//
// Not retained:
//   f2  C99 inline definition
//   f3  non-inline
//   f4  GNU C89/C90 extern inline

static inline int f1(int x) { return x + 1; }

inline int f2(int x) { return x + 2; }

int f3(int x) { return x + 3; }

__attribute__((gnu_inline)) extern inline int f4(int x) { return x + 4; }

extern inline int f5(int x);
inline int f5(int x) { return x + 5; }

// f1 and f5 are unused, so their definitions are emitted due to -fkeep-inline-functions.

// CHECK: @llvm{{(\.compiler)?}}.used = appending global [2 x ptr] [ptr @f1, ptr @f5]
// CHECK: define internal {{.*}}@f1
// CHECK: define {{.*}}@f5

int use(void) { return  f2(0) + f3(0) + f4(0); }

// CHECK-GNU89: @llvm{{(\.compiler)?}}.used = appending global [3 x ptr] [ptr @f1, ptr @f2, ptr @f5]
// CHECK-GNU89: define {{.*}}@f1
// CHECK-GNU89: define {{.*}}@f2
// CHECK-GNU89: define {{.*}}@f5
