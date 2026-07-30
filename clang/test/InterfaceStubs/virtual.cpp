// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck -check-prefix=CHECK-TAPI %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck -check-prefix=CHECK-TAPI2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | \
// RUN: llvm-readelf -s - 2>&1 | FileCheck -check-prefix=CHECK-SYMBOLS %s

#define HIDDEN  __attribute__((__visibility__(("hidden"))))
#define DEFAULT __attribute__((__visibility__(("default"))))

// CHECK-TAPI-NOT: _ZNK1Q5func1Ev
// CHECK-TAPI-NOT: _ZNK1Q5func2Ev
// CHECK-SYMBOLS-DAG: NOTYPE  GLOBAL HIDDEN   {{.*}} _ZNK1Q5func1Ev
// CHECK-SYMBOLS-DAG: NOTYPE  GLOBAL DEFAULT  {{.*}} _ZNK1Q5func2Ev
struct Q {
  virtual HIDDEN  int func1() const;
  virtual DEFAULT int func2() const;
} q;

// CHECK-TAPI-NOT: _ZNK1S5func1Ev
// CHECK-TAPI2-DAG: _ZNK1S5func2Ev
// CHECK-SYMBOLS-DAG: FUNC    WEAK   HIDDEN   {{.*}} _ZNK1S5func1Ev
// CHECK-SYMBOLS-DAG: FUNC    WEAK   DEFAULT  {{.*}} _ZNK1S5func2Ev
struct S {
  virtual HIDDEN  int func1() const { return 42; }
  virtual DEFAULT int func2() const { return 42; }
} s;

// CHECK-TAPI-NOT: _ZNK1R5func1Ev
// CHECK-TAPI-NOT: _ZNK1R5func2Ev
// CHECK-SYMBOLS-NOT: _ZNK1R5func1Ev
// CHECK-SYMBOLS-NOT: _ZNK1R5func2Ev
struct R {
  virtual HIDDEN  int func1() const = 0;
  virtual DEFAULT int func2() const = 0;
};

int a = q.func1() + q.func2();

