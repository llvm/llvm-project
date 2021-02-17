// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DPARENT_CLASS_VISIBILITY="" -DCHILD_CLASS_VISIBILITY="" \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: FileCheck -check-prefix=CHECK-X %s
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: -DPARENT_CLASS_VISIBILITY="" -DCHILD_CLASS_VISIBILITY="" \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: llvm-readelf -s - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-X-RE %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DPARENT_CLASS_VISIBILITY=HIDDEN -DCHILD_CLASS_VISIBILITY="" \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: FileCheck -check-prefix=CHECK-HP %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-ifs-v2 \
// RUN: -DPARENT_CLASS_VISIBILITY=HIDDEN -DCHILD_CLASS_VISIBILITY="" \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: FileCheck -check-prefix=CHECK-HP2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: -DPARENT_CLASS_VISIBILITY=HIDDEN -DCHILD_CLASS_VISIBILITY="" \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: llvm-readelf -s - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-HP-RE %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DPARENT_CLASS_VISIBILITY="" -DCHILD_CLASS_VISIBILITY=HIDDEN \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: FileCheck -check-prefix=CHECK-HC %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DPARENT_CLASS_VISIBILITY="" -DCHILD_CLASS_VISIBILITY=HIDDEN \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: FileCheck -check-prefix=CHECK-HC2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: -DPARENT_CLASS_VISIBILITY="" -DCHILD_CLASS_VISIBILITY=HIDDEN \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: llvm-readelf -s - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-HC-RE %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DPARENT_CLASS_VISIBILITY=HIDDEN -DCHILD_CLASS_VISIBILITY=HIDDEN \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: FileCheck -check-prefix=CHECK-HP-HC %s
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: -DPARENT_CLASS_VISIBILITY=HIDDEN -DCHILD_CLASS_VISIBILITY=HIDDEN \
// RUN: -DPARENT_METHOD_VISIBILITY="" -DCHILD_METHOD_VISIBILITY="" %s | \
// RUN: llvm-readelf -s - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-HP-HC-RE %s

// CHECK-X-DAG: _ZN1CC2Ev
// CHECK-X-DAG: _ZN1CD0Ev
// CHECK-X-DAG: _ZN1CD2Ev
// CHECK-X-DAG: _ZN1SC2Ev
// CHECK-X-DAG: _ZN1SD0Ev
// CHECK-X-DAG: _ZN1SD2Ev
// CHECK-X-DAG: _ZN1C1mEv
// CHECK-X-DAG: _ZN1S1nEv

// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1C1mEv
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1CC2Ev
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1CD0Ev
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1CD2Ev
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1S1nEv
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1SC2Ev
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1SD0Ev
// CHECK-X-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1SD2Ev

// CHECK-HP2-DAG: _ZN1CC2Ev
// CHECK-HP2-DAG: _ZN1CD0Ev
// CHECK-HP2-DAG: _ZN1CD2Ev
// CHECK-HP2-DAG: _ZN1C1mEv

// CHECK-HP-NOT: _ZN1S1nEv
// CHECK-HP-NOT: _ZN1SC2Ev
// CHECK-HP-NOT: _ZN1SD0Ev
// CHECK-HP-NOT: _ZN1SD2Ev

// CHECK-HP-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1C1mEv
// CHECK-HP-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1CC2Ev
// CHECK-HP-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1CD0Ev
// CHECK-HP-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1CD2Ev
// CHECK-HP-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1S1nEv
// CHECK-HP-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1SC2Ev
// CHECK-HP-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1SD0Ev
// CHECK-HP-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1SD2Ev

// CHECK-HC2-DAG: _ZN1SC2Ev
// CHECK-HC2-DAG: _ZN1SD0Ev
// CHECK-HC2-DAG: _ZN1SD2Ev
// CHECK-HC2-DAG: _ZN1S1nEv

// CHECK-HC-NOT: _ZN1C1mEv
// CHECK-HC-NOT: _ZN1CC2Ev
// CHECK-HC-NOT: _ZN1CD0Ev
// CHECK-HC-NOT: _ZN1CD2Ev

// CHECK-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1C1mEv
// CHECK-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1CC2Ev
// CHECK-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1CD0Ev
// CHECK-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1CD2Ev
// CHECK-HC-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1S1nEv
// CHECK-HC-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1SC2Ev
// CHECK-HC-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1SD0Ev
// CHECK-HC-RE-DAG: FUNC    WEAK   DEFAULT   [[#]] _ZN1SD2Ev

// CHECK-HP-HC-NOT: _ZN1CC2Ev
// CHECK-HP-HC-NOT: _ZN1CD0Ev
// CHECK-HP-HC-NOT: _ZN1CD2Ev
// CHECK-HP-HC-NOT: _ZN1SC2Ev
// CHECK-HP-HC-NOT: _ZN1SD0Ev
// CHECK-HP-HC-NOT: _ZN1SD2Ev
// CHECK-HP-HC-NOT: _ZN1C1mEv
// CHECK-HP-HC-NOT: _ZN1S1nEv

// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1C1mEv
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1CC2Ev
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1CD0Ev
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1CD2Ev
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1S1nEv
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1SC2Ev
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1SD0Ev
// CHECK-HP-HC-RE-DAG: FUNC    WEAK   HIDDEN    [[#]] _ZN1SD2Ev

// TODO: clang+llvm does not materialize complete ctors and dtors for the
// Itanium abi. Figure out why and add the check-not for these:
// _ZN1CC1Ev
// _ZN1CD1Ev
// _ZN1SC1Ev
// _ZN1SD1Ev

#define HIDDEN __attribute__((__visibility__("hidden")))
#define DEFAULT __attribute__((__visibility__("default")))

struct PARENT_CLASS_VISIBILITY S {
  virtual ~S() {}
  virtual PARENT_METHOD_VISIBILITY void n() {}
};

class CHILD_CLASS_VISIBILITY C : public S {
public:
  virtual CHILD_METHOD_VISIBILITY void m() {}
};

void f() {
  C c;
  c.m();
  c.n();
}
