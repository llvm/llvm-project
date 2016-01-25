// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM --check-prefix=NDIAG %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM --check-prefix=DIAG --check-prefix=DIAG-ABORT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vcall -fsanitize-recover=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM --check-prefix=DIAG --check-prefix=DIAG-RECOVER %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS --check-prefix=NDIAG %s

// MS: @[[VTA:[0-9]*]] {{.*}} comdat($"\01??_7A@@6B@")
// MS: @[[VTB:[0-9]*]] {{.*}} comdat($"\01??_7B@@6B0@@")
// MS: @[[VTAinB:[0-9]*]] {{.*}} comdat($"\01??_7B@@6BA@@@")
// MS: @[[VTAinC:[0-9]*]] {{.*}} comdat($"\01??_7C@@6B@")
// MS: @[[VTBinD:[0-9]*]] {{.*}} comdat($"\01??_7D@?A@@6BB@@@")
// MS: @[[VTAinBinD:[0-9]*]] {{.*}} comdat($"\01??_7D@?A@@6BA@@@")
// MS: @[[VTFA:[0-9]*]] {{.*}} comdat($"\01??_7FA@?1??foo@@YAXXZ@6B@")

struct A {
  A();
  virtual void f();
};

struct B : virtual A {
  B();
  virtual void g();
  virtual void h();
};

struct C : virtual A {
  C();
};

namespace {

struct D : B, C {
  D();
  virtual void f();
  virtual void h();
};

}

A::A() {}
B::B() {}
C::C() {}
D::D() {}

void A::f() {
}

void B::g() {
}

void D::f() {
}

void D::h() {
}

// DIAG: @[[SRC:.*]] = private unnamed_addr constant [{{.*}} x i8] c"{{.*}}cfi-vcall.cpp\00", align 1
// DIAG: @[[TYPE:.*]] = private unnamed_addr constant { i16, i16, [4 x i8] } { i16 -1, i16 0, [4 x i8] c"'A'\00" }
// DIAG: @[[BADTYPESTATIC:.*]] = private unnamed_addr global { i8, { [{{.*}} x i8]*, i32, i32 }, { i16, i16, [4 x i8] }* } { i8 0, { [{{.*}} x i8]*, i32, i32 } { [{{.*}} x i8]* @[[SRC]], i32 [[@LINE+21]], i32 3 }, { i16, i16, [4 x i8] }* @[[TYPE]] }

// ITANIUM: define void @_Z2afP1A
// MS: define void @"\01?af@@YAXPEAUA@@@Z"
void af(A *a) {
  // ITANIUM: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* [[VT:%[^ ]*]], metadata !"_ZTS1A")
  // MS: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* [[VT:%[^ ]*]], metadata !"?AUA@@")
  // CHECK-NEXT: br i1 [[P]], label %[[CONTBB:[^ ,]*]], label %[[TRAPBB:[^ ,]*]]
  // CHECK-NEXT: {{^$}}

  // CHECK: [[TRAPBB]]
  // NDIAG-NEXT: call void @llvm.trap()
  // NDIAG-NEXT: unreachable
  // DIAG-NEXT: [[VTINT:%[^ ]*]] = ptrtoint i8* [[VT]] to i64
  // DIAG-ABORT-NEXT: call void @__ubsan_handle_cfi_check_fail_abort(i8* getelementptr inbounds ({{.*}} @[[BADTYPESTATIC]], i32 0, i32 0), i64 [[VTINT]])
  // DIAG-ABORT-NEXT: unreachable
  // DIAG-RECOVER-NEXT: call void @__ubsan_handle_cfi_check_fail(i8* getelementptr inbounds ({{.*}} @[[BADTYPESTATIC]], i32 0, i32 0), i64 [[VTINT]])
  // DIAG-RECOVER-NEXT: br label %[[CONTBB]]

  // CHECK: [[CONTBB]]
  // CHECK: call void %
  a->f();
}

// ITANIUM: define internal void @_Z3df1PN12_GLOBAL__N_11DE
// MS: define internal void @"\01?df1@@YAXPEAUD@?A@@@Z"
void df1(D *d) {
  // ITANIUM: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata ![[DTYPE:[0-9]+]])
  // MS: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"?AUA@@")
  d->f();
}

// ITANIUM: define internal void @_Z3dg1PN12_GLOBAL__N_11DE
// MS: define internal void @"\01?dg1@@YAXPEAUD@?A@@@Z"
void dg1(D *d) {
  // ITANIUM: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // MS: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"?AUB@@")
  d->g();
}

// ITANIUM: define internal void @_Z3dh1PN12_GLOBAL__N_11DE
// MS: define internal void @"\01?dh1@@YAXPEAUD@?A@@@Z"
void dh1(D *d) {
  // ITANIUM: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata ![[DTYPE]])
  // MS: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata ![[DTYPE:[0-9]+]])
  d->h();
}

// ITANIUM: define internal void @_Z3df2PN12_GLOBAL__N_11DE
// MS: define internal void @"\01?df2@@YAXPEAUD@?A@@@Z"
__attribute__((no_sanitize("cfi")))
void df2(D *d) {
  // CHECK-NOT: call i1 @llvm.bitset.test
  d->f();
}

// ITANIUM: define internal void @_Z3df3PN12_GLOBAL__N_11DE
// MS: define internal void @"\01?df3@@YAXPEAUD@?A@@@Z"
__attribute__((no_sanitize("address"))) __attribute__((no_sanitize("cfi-vcall")))
void df3(D *d) {
  // CHECK-NOT: call i1 @llvm.bitset.test
  d->f();
}

D d;

void foo() {
  df1(&d);
  dg1(&d);
  dh1(&d);
  df2(&d);
  df3(&d);

  struct FA : A {
    void f() {}
  } fa;
  af(&fa);
}

namespace test2 {

struct A {
  virtual void m_fn1();
};
struct B {
  virtual void m_fn2();
};
struct C : B, A {};
struct D : C {
  void m_fn1();
};

// ITANIUM: define void @_ZN5test21fEPNS_1DE
// MS: define void @"\01?f@test2@@YAXPEAUD@1@@Z"
void f(D *d) {
  // ITANIUM: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"_ZTSN5test21DE")
  // MS: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"?AUA@test2@@")
  d->m_fn1();
}

}

// Check for the expected number of elements (9 or 15 respectively).
// MS: !llvm.bitsets = !{[[X:[^,]*(,[^,]*){8}]]}
// ITANIUM: !llvm.bitsets = !{[[X:[^,]*(,[^,]*){14}]]}

// ITANIUM-DAG: !{!"_ZTS1A", [3 x i8*]* @_ZTV1A, i64 16}
// ITANIUM-DAG: !{!"_ZTS1A", [7 x i8*]* @_ZTCN12_GLOBAL__N_11DE0_1B, i64 32}
// ITANIUM-DAG: !{!"_ZTS1B", [7 x i8*]* @_ZTCN12_GLOBAL__N_11DE0_1B, i64 32}
// ITANIUM-DAG: !{!"_ZTS1A", [9 x i8*]* @_ZTCN12_GLOBAL__N_11DE8_1C, i64 64}
// ITANIUM-DAG: !{!"_ZTS1C", [9 x i8*]* @_ZTCN12_GLOBAL__N_11DE8_1C, i64 32}
// ITANIUM-DAG: !{!"_ZTS1A", [12 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// ITANIUM-DAG: !{!"_ZTS1B", [12 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// ITANIUM-DAG: !{!"_ZTS1C", [12 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 88}
// ITANIUM-DAG: !{![[DTYPE]], [12 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// ITANIUM-DAG: !{!"_ZTS1A", [7 x i8*]* @_ZTV1B, i64 32}
// ITANIUM-DAG: !{!"_ZTS1B", [7 x i8*]* @_ZTV1B, i64 32}
// ITANIUM-DAG: !{!"_ZTS1A", [5 x i8*]* @_ZTV1C, i64 32}
// ITANIUM-DAG: !{!"_ZTS1C", [5 x i8*]* @_ZTV1C, i64 32}
// ITANIUM-DAG: !{!"_ZTS1A", [3 x i8*]* @_ZTVZ3foovE2FA, i64 16}
// ITANIUM-DAG: !{!{{[0-9]+}}, [3 x i8*]* @_ZTVZ3foovE2FA, i64 16}

// MS-DAG: !{!"?AUA@@", [2 x i8*]* @[[VTA]], i64 8}
// MS-DAG: !{!"?AUB@@", [3 x i8*]* @[[VTB]], i64 8}
// MS-DAG: !{!"?AUA@@", [2 x i8*]* @[[VTAinB]], i64 8}
// MS-DAG: !{!"?AUA@@", [2 x i8*]* @[[VTAinC]], i64 8}
// MS-DAG: !{!"?AUB@@", [3 x i8*]* @[[VTBinD]], i64 8}
// MS-DAG: !{![[DTYPE]], [3 x i8*]* @[[VTBinD]], i64 8}
// MS-DAG: !{!"?AUA@@", [2 x i8*]* @[[VTAinBinD]], i64 8}
// MS-DAG: !{!"?AUA@@", [2 x i8*]* @[[VTFA]], i64 8}
// MS-DAG: !{!{{[0-9]+}}, [2 x i8*]* @[[VTFA]], i64 8}
