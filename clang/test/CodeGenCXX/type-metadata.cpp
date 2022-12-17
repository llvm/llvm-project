// Tests for the cfi-vcall feature:
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT-NO-RV --check-prefix=ITANIUM --check-prefix=ITANIUM-MD --check-prefix=TT-ITANIUM-HIDDEN --check-prefix=NDIAG %s
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT-NO-RV --check-prefix=ITANIUM --check-prefix=ITANIUM-MD --check-prefix=TT-ITANIUM-HIDDEN --check-prefix=ITANIUM-MD-DIAG --check-prefix=ITANIUM-DIAG --check-prefix=DIAG --check-prefix=DIAG-ABORT %s
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -fsanitize-recover=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT-NO-RV --check-prefix=ITANIUM --check-prefix=ITANIUM-MD --check-prefix=TT-ITANIUM-HIDDEN --check-prefix=ITANIUM-MD-DIAG --check-prefix=ITANIUM-DIAG --check-prefix=DIAG --check-prefix=DIAG-RECOVER %s
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-pc-windows-msvc -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT-NO-RV --check-prefix=MS --check-prefix=TT-MS --check-prefix=NDIAG %s

// Tests for the whole-program-vtables feature:
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=VTABLE-OPT --check-prefix=ITANIUM --check-prefix=ITANIUM-MD --check-prefix=TT-ITANIUM-HIDDEN %s
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=VTABLE-OPT --check-prefix=ITANIUM-DEFAULTVIS --check-prefix=TT-ITANIUM-DEFAULT %s
// RUN: %clang_cc1 -O2 -flto -flto-unit -triple x86_64-unknown-linux -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=ITANIUM-OPT --check-prefix=ITANIUM-OPT-LAYOUT %s
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-pc-windows-msvc -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=VTABLE-OPT --check-prefix=MS --check-prefix=TT-MS %s

// Tests for cfi + whole-program-vtables:
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-VT --check-prefix=ITANIUM --check-prefix=TC-ITANIUM --check-prefix=ITANIUM-MD %s
// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-pc-windows-msvc -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-VT --check-prefix=MS --check-prefix=TC-MS %s

// Equivalent tests for above, but with relative-vtables.
// Tests for the cfi-vcall feature:
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT --check-prefix=RV-MD --check-prefix=ITANIUM --check-prefix=TT-ITANIUM-HIDDEN --check-prefix=NDIAG --check-prefix=CFI-NVT-RV %s
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT --check-prefix=CFI-NVT-RV --check-prefix=ITANIUM --check-prefix=RV-MD --check-prefix=TT-ITANIUM-HIDDEN --check-prefix=ITANIUM-DIAG --check-prefix=RV-MD-DIAG --check-prefix=DIAG --check-prefix=DIAG-ABORT %s
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -fsanitize-recover=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-NVT --check-prefix=CFI-NVT-RV --check-prefix=ITANIUM --check-prefix=RV-MD --check-prefix=TT-ITANIUM-HIDDEN --check-prefix=ITANIUM-DIAG --check-prefix=RV-MD-DIAG --check-prefix=DIAG --check-prefix=DIAG-RECOVER %s

// Tests for the whole-program-vtables feature:
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=VTABLE-OPT --check-prefix=ITANIUM -check-prefix=RV-MD --check-prefix=TT-ITANIUM-HIDDEN %s
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -flto -flto-unit -triple x86_64-unknown-linux -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=VTABLE-OPT --check-prefix=ITANIUM-DEFAULTVIS --check-prefix=TT-ITANIUM-DEFAULT %s
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -O2 -flto -flto-unit -triple x86_64-unknown-linux -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=ITANIUM-OPT --check-prefix=RV-OPT-LAYOUT %s

// Tests for cfi + whole-program-vtables:
// RUN: %clang_cc1 -fexperimental-relative-c++-abi-vtables -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=CFI --check-prefix=CFI-VT --check-prefix=ITANIUM --check-prefix=RV-MD --check-prefix=TC-ITANIUM %s

// ITANIUM: @_ZTV1A = {{[^!]*}}, !type [[A16:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL16:![0-9]+]]
// ITANIUM-SAME: !type [[AF16:![0-9]+]]

// ITANIUM: @_ZTV1B = {{[^!]*}}, !type [[A32:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL32:![0-9]+]]
// ITANIUM-SAME: !type [[AF32:![0-9]+]]
// ITANIUM-SAME: !type [[AF40:![0-9]+]]
// ITANIUM-SAME: !type [[AF48:![0-9]+]]
// ITANIUM-SAME: !type [[B32:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[BF32:![0-9]+]]
// ITANIUM-SAME: !type [[BF40:![0-9]+]]
// ITANIUM-SAME: !type [[BF48:![0-9]+]]

// ITANIUM: @_ZTV1C = {{[^!]*}}, !type [[A32]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[AF32]]
// ITANIUM-SAME: !type [[C32:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[CF32:![0-9]+]]

// DIAG: @[[SRC:.*]] = private unnamed_addr constant [{{.*}} x i8] c"{{.*}}type-metadata.cpp\00", align 1
// DIAG: @[[TYPE:.*]] = private unnamed_addr constant { i16, i16, [4 x i8] } { i16 -1, i16 0, [4 x i8] c"'A'\00" }
// DIAG: @[[BADTYPESTATIC:.*]] = private unnamed_addr global { i8, { ptr, i32, i32 }, ptr } { i8 0, { ptr, i32, i32 } { ptr @[[SRC]], i32 123, i32 3 }, ptr @[[TYPE]] }

// ITANIUM: @_ZTVN12_GLOBAL__N_11DE = {{[^!]*}}, !type [[A32]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[AF32]]
// ITANIUM-SAME: !type [[AF40]]
// ITANIUM-SAME: !type [[AF48]]
// ITANIUM-SAME: !type [[B32]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[BF32]]
// ITANIUM-SAME: !type [[BF40]]
// ITANIUM-SAME: !type [[BF48]]
// ITANIUM-SAME: !type [[C88:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL88:![0-9]+]]
// ITANIUM-SAME: !type [[CF32]]
// ITANIUM-SAME: !type [[CF40:![0-9]+]]
// ITANIUM-SAME: !type [[CF48:![0-9]+]]
// ITANIUM-SAME: !type [[D32:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[DF32:![0-9]+]]
// ITANIUM-SAME: !type [[DF40:![0-9]+]]
// ITANIUM-SAME: !type [[DF48:![0-9]+]]

// ITANIUM: @_ZTCN12_GLOBAL__N_11DE0_1B = {{[^!]*}}, !type [[A32]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[B32]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]

// ITANIUM: @_ZTCN12_GLOBAL__N_11DE8_1C = {{[^!]*}}, !type [[A64:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL64:![0-9]+]]
// ITANIUM-SAME: !type [[AF64:![0-9]+]]
// ITANIUM-SAME: !type [[C32]]
// ITANIUM-DIAG-SAME: !type [[ALL32]]
// ITANIUM-SAME: !type [[CF64:![0-9]+]]

// ITANIUM: @_ZTVZ3foovE2FA = {{[^!]*}}, !type [[A16]]
// ITANIUM-DIAG-SAME: !type [[ALL16]]
// ITANIUM-SAME: !type [[AF16]]
// ITANIUM-SAME: !type [[FA16:![0-9]+]]
// ITANIUM-DIAG-SAME: !type [[ALL16]]
// ITANIUM-SAME: !type [[FAF16:![0-9]+]]

// ITANIUM: @_ZTVN5test31EE = external unnamed_addr constant
// ITANIUM-DEFAULTVIS: @_ZTVN5test31EE = external unnamed_addr constant
// ITANIUM-OPT: @_ZTVN5test31EE = available_externally unnamed_addr constant {{[^!]*}},
// ITANIUM-OPT-SAME: !type [[E16:![0-9]+]],
// ITANIUM-OPT-SAME: !type [[EF16:![0-9]+]]
// ITANIUM-OPT: @llvm.compiler.used = appending global [1 x ptr] [ptr @_ZTVN5test31EE]

// MS: comdat($"??_7A@@6B@"), !type [[A8:![0-9]+]]
// MS: comdat($"??_7B@@6B0@@"), !type [[B8:![0-9]+]]
// MS: comdat($"??_7B@@6BA@@@"), !type [[A8]]
// MS: comdat($"??_7C@@6B@"), !type [[A8]]
// MS: comdat($"??_7D@?A0x{{[^@]*}}@@6BB@@@"), !type [[B8]], !type [[D8:![0-9]+]]
// MS: comdat($"??_7D@?A0x{{[^@]*}}@@6BA@@@"), !type [[A8]]
// MS: comdat($"??_7FA@?1??foo@@YAXXZ@6B@"), !type [[A8]], !type [[FA8:![0-9]+]]

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

// ITANIUM: define hidden void @_Z2afP1A
// ITANIUM-DEFAULTVIS: define{{.*}} void @_Z2afP1A
// MS: define dso_local void @"?af@@YAXPEAUA@@@Z"
void af(A *a) {
  // TT-ITANIUM-HIDDEN: [[P:%[^ ]*]] = call i1 @llvm.type.test(ptr [[VT:%[^ ]*]], metadata !"_ZTS1A")
  // TT-ITANIUM-DEFAULT: [[P:%[^ ]*]] = call i1 @llvm.public.type.test(ptr [[VT:%[^ ]*]], metadata !"_ZTS1A")
  // TT-MS: [[P:%[^ ]*]] = call i1 @llvm.type.test(ptr [[VT:%[^ ]*]], metadata !"?AUA@@")
  // TC-ITANIUM: [[PAIR:%[^ ]*]] = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 0, metadata !"_ZTS1A")
  // TC-MS: [[PAIR:%[^ ]*]] = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 0, metadata !"?AUA@@")
  // CFI-VT: [[P:%[^ ]*]] = extractvalue { ptr, i1 } [[PAIR]], 1
  // DIAG-NEXT: [[VTVALID0:%[^ ]*]] = call i1 @llvm.type.test(ptr [[VT]], metadata !"all-vtables")
  // VTABLE-OPT: call void @llvm.assume(i1 [[P]])
  // CFI-NEXT: br i1 [[P]], label %[[CONTBB:[^ ,]*]], label %[[TRAPBB:[^ ,]*]]
  // CFI-NEXT: {{^$}}

  // CFI: [[TRAPBB]]
  // NDIAG-NEXT: call void @llvm.ubsantrap(i8 2)
  // NDIAG-NEXT: unreachable
  // DIAG-NEXT: [[VTINT:%[^ ]*]] = ptrtoint ptr [[VT]] to i64
  // DIAG-NEXT: [[VTVALID:%[^ ]*]] = zext i1 [[VTVALID0]] to i64
  // DIAG-ABORT-NEXT: call void @__ubsan_handle_cfi_check_fail_abort(ptr @[[BADTYPESTATIC]], i64 [[VTINT]], i64 [[VTVALID]])
  // DIAG-ABORT-NEXT: unreachable
  // DIAG-RECOVER-NEXT: call void @__ubsan_handle_cfi_check_fail(ptr @[[BADTYPESTATIC]], i64 [[VTINT]], i64 [[VTVALID]])
  // DIAG-RECOVER-NEXT: br label %[[CONTBB]]

  // CFI: [[CONTBB]]
  // CFI-NVT-NO-RV: [[PTR:%[^ ]*]] = load
  // CFI-NVT-RV: [[PTR:%[^ ]*]] = call ptr @llvm.load.relative.i32
  // CFI-VT: [[PTR:%[^ ]*]] = extractvalue { ptr, i1 } [[PAIR]], 0
  // CFI: call void [[PTR]]
#line 123
  a->f();
}

// ITANIUM: define internal void @_Z3df1PN12_GLOBAL__N_11DE
// MS: define internal void @"?df1@@YAXPEAUD@?A0x{{[^@]*}}@@@Z"
void df1(D *d) {
  // TT-ITANIUM-HIDDEN: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata ![[DTYPE:[0-9]+]])
  // TT-ITANIUM-DEFAULT: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata ![[DTYPE:[0-9]+]])
  // TT-MS: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata !"?AUA@@")
  // TC-ITANIUM: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 0, metadata ![[DTYPE:[0-9]+]])
  // TC-MS: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 0, metadata !"?AUA@@")
  d->f();
}

// ITANIUM: define internal void @_Z3dg1PN12_GLOBAL__N_11DE
// MS: define internal void @"?dg1@@YAXPEAUD@?A0x{{[^@]*}}@@@Z"
void dg1(D *d) {
  // TT-ITANIUM-HIDDEN: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata !"_ZTS1B")
  // TT-ITANIUM-DEFAULT: {{%[^ ]*}} = call i1 @llvm.public.type.test(ptr {{%[^ ]*}}, metadata !"_ZTS1B")
  // TT-MS: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata !"?AUB@@")
  // TC-ITANIUM: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 8, metadata !"_ZTS1B")
  // TC-MS: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 0, metadata !"?AUB@@")
  d->g();
}

// ITANIUM: define internal void @_Z3dh1PN12_GLOBAL__N_11DE
// MS: define internal void @"?dh1@@YAXPEAUD@?A0x{{[^@]*}}@@@Z"
void dh1(D *d) {
  // TT-ITANIUM-HIDDEN: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata ![[DTYPE]])
  // TT-ITANIUM-DEFAULT: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata ![[DTYPE]])
  // TT-MS: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata ![[DTYPE:[0-9]+]])
  // TC-ITANIUM: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 16, metadata ![[DTYPE]])
  // TC-MS: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 8, metadata ![[DTYPE:[0-9]+]])
  d->h();
}

// ITANIUM: define internal void @_Z3df2PN12_GLOBAL__N_11DE
// MS: define internal void @"?df2@@YAXPEAUD@?A0x{{[^@]*}}@@@Z"
__attribute__((no_sanitize("cfi")))
void df2(D *d) {
  // CFI-NVT-NOT: call i1 @llvm.type.test
  // CFI-VT: [[P:%[^ ]*]] = call i1 @llvm.type.test
  // CFI-VT: call void @llvm.assume(i1 [[P]])
  d->f();
}

// ITANIUM: define internal void @_Z3df3PN12_GLOBAL__N_11DE
// MS: define internal void @"?df3@@YAXPEAUD@?A0x{{[^@]*}}@@@Z"
__attribute__((no_sanitize("address"))) __attribute__((no_sanitize("cfi-vcall")))
void df3(D *d) {
  // CFI-NVT-NOT: call i1 @llvm.type.test
  // CFI-VT: [[P:%[^ ]*]] = call i1 @llvm.type.test
  // CFI-VT: call void @llvm.assume(i1 [[P]])
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

// ITANIUM: define hidden void @_ZN5test21fEPNS_1DE
// ITANIUM-DEFAULTVIS: define{{.*}} void @_ZN5test21fEPNS_1DE
// MS: define dso_local void @"?f@test2@@YAXPEAUD@1@@Z"
void f(D *d) {
  // TT-ITANIUM-HIDDEN: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata !"_ZTSN5test21DE")
  // TT-ITANIUM-DEFAULT: {{%[^ ]*}} = call i1 @llvm.public.type.test(ptr {{%[^ ]*}}, metadata !"_ZTSN5test21DE")
  // TT-MS: {{%[^ ]*}} = call i1 @llvm.type.test(ptr {{%[^ ]*}}, metadata !"?AUA@test2@@")
  // TC-ITANIUM: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 8, metadata !"_ZTSN5test21DE")
  // TC-MS: {{%[^ ]*}} = call { ptr, i1 } @llvm.type.checked.load(ptr {{%[^ ]*}}, i32 0, metadata !"?AUA@test2@@")
  d->m_fn1();
}

}

namespace test3 {
// All virtual functions are outline, so we can assume that it will
// be generated in translation unit where foo is defined.
struct E {
  virtual void foo();
};

void g() {
  E e;
  e.foo();
}

}  // Test9

// RV-MD: [[A16]] = !{i64 8, !"_ZTS1A"}
// RV-MD-DIAG: [[ALL16]] = !{i64 8, !"all-vtables"}
// RV-MD: [[AF16]] = !{i64 8, !"_ZTSM1AFvvE.virtual"}
// RV-MD: [[A32]] = !{i64 16, !"_ZTS1A"}
// RV-MD-DIAG: [[ALL32]] = !{i64 16, !"all-vtables"}
// RV-MD: [[AF32]] = !{i64 16, !"_ZTSM1AFvvE.virtual"}
// RV-MD: [[AF40]] = !{i64 20, !"_ZTSM1AFvvE.virtual"}
// RV-MD: [[AF48]] = !{i64 24, !"_ZTSM1AFvvE.virtual"}
// RV-MD: [[B32]] = !{i64 16, !"_ZTS1B"}
// RV-MD: [[BF32]] = !{i64 16, !"_ZTSM1BFvvE.virtual"}
// RV-MD: [[BF40]] = !{i64 20, !"_ZTSM1BFvvE.virtual"}
// RV-MD: [[BF48]] = !{i64 24, !"_ZTSM1BFvvE.virtual"}
// RV-MD: [[C32]] = !{i64 16, !"_ZTS1C"}
// RV-MD: [[CF32]] = !{i64 16, !"_ZTSM1CFvvE.virtual"}
// RV-MD: [[C88]] = !{i64 44, !"_ZTS1C"}
// RV-MD-DIAG: [[ALL88]] = !{i64 44, !"all-vtables"}
// RV-MD: [[CF40]] = !{i64 20, !"_ZTSM1CFvvE.virtual"}
// RV-MD: [[CF48]] = !{i64 24, !"_ZTSM1CFvvE.virtual"}
// RV-MD: [[D32]] = !{i64 16, [[D_ID:![0-9]+]]}
// RV-MD: [[D_ID]] = distinct !{}
// RV-MD: [[DF32]] = !{i64 16, [[DF_ID:![0-9]+]]}
// RV-MD: [[DF_ID]] = distinct !{}
// RV-MD: [[DF40]] = !{i64 20, [[DF_ID]]}
// RV-MD: [[DF48]] = !{i64 24, [[DF_ID]]}
// RV-MD: [[A64]] = !{i64 32, !"_ZTS1A"}
// RV-MD-DIAG: [[ALL64]] = !{i64 32, !"all-vtables"}
// RV-MD: [[AF64]] = !{i64 32, !"_ZTSM1AFvvE.virtual"}
// RV-MD: [[CF64]] = !{i64 32, !"_ZTSM1CFvvE.virtual"}
// RV-MD: [[FA16]] = !{i64 8, [[FA_ID:![0-9]+]]}
// RV-MD: [[FA_ID]] = distinct !{}
// RV-MD: [[FAF16]] = !{i64 8, [[FAF_ID:![0-9]+]]}
// RV-MD: [[FAF_ID]] = distinct !{}

// ITANIUM-MD: [[A16]] = !{i64 16, !"_ZTS1A"}
// ITANIUM-MD-DIAG: [[ALL16]] = !{i64 16, !"all-vtables"}
// ITANIUM-MD: [[AF16]] = !{i64 16, !"_ZTSM1AFvvE.virtual"}
// ITANIUM-MD: [[A32]] = !{i64 32, !"_ZTS1A"}
// ITANIUM-MD-DIAG: [[ALL32]] = !{i64 32, !"all-vtables"}
// ITANIUM-MD: [[AF32]] = !{i64 32, !"_ZTSM1AFvvE.virtual"}
// ITANIUM-MD: [[AF40]] = !{i64 40, !"_ZTSM1AFvvE.virtual"}
// ITANIUM-MD: [[AF48]] = !{i64 48, !"_ZTSM1AFvvE.virtual"}
// ITANIUM-MD: [[B32]] = !{i64 32, !"_ZTS1B"}
// ITANIUM-MD: [[BF32]] = !{i64 32, !"_ZTSM1BFvvE.virtual"}
// ITANIUM-MD: [[BF40]] = !{i64 40, !"_ZTSM1BFvvE.virtual"}
// ITANIUM-MD: [[BF48]] = !{i64 48, !"_ZTSM1BFvvE.virtual"}
// ITANIUM-MD: [[C32]] = !{i64 32, !"_ZTS1C"}
// ITANIUM-MD: [[CF32]] = !{i64 32, !"_ZTSM1CFvvE.virtual"}
// ITANIUM-MD: [[C88]] = !{i64 88, !"_ZTS1C"}
// ITANIUM-MD-DIAG: [[ALL88]] = !{i64 88, !"all-vtables"}
// ITANIUM-MD: [[CF40]] = !{i64 40, !"_ZTSM1CFvvE.virtual"}
// ITANIUM-MD: [[CF48]] = !{i64 48, !"_ZTSM1CFvvE.virtual"}
// ITANIUM-MD: [[D32]] = !{i64 32, [[D_ID:![0-9]+]]}
// ITANIUM-MD: [[D_ID]] = distinct !{}
// ITANIUM-MD: [[DF32]] = !{i64 32, [[DF_ID:![0-9]+]]}
// ITANIUM-MD: [[DF_ID]] = distinct !{}
// ITANIUM-MD: [[DF40]] = !{i64 40, [[DF_ID]]}
// ITANIUM-MD: [[DF48]] = !{i64 48, [[DF_ID]]}
// ITANIUM-MD: [[A64]] = !{i64 64, !"_ZTS1A"}
// ITANIUM-MD-DIAG: [[ALL64]] = !{i64 64, !"all-vtables"}
// ITANIUM-MD: [[AF64]] = !{i64 64, !"_ZTSM1AFvvE.virtual"}
// ITANIUM-MD: [[CF64]] = !{i64 64, !"_ZTSM1CFvvE.virtual"}
// ITANIUM-MD: [[FA16]] = !{i64 16, [[FA_ID:![0-9]+]]}
// ITANIUM-MD: [[FA_ID]] = distinct !{}
// ITANIUM-MD: [[FAF16]] = !{i64 16, [[FAF_ID:![0-9]+]]}
// ITANIUM-MD: [[FAF_ID]] = distinct !{}

// ITANIUM-OPT-LAYOUT: [[E16]] = !{i64 16, !"_ZTSN5test31EE"}
// ITANIUM-OPT-LAYOUT: [[EF16]] = !{i64 16, !"_ZTSMN5test31EEFvvE.virtual"}
// RV-OPT-LAYOUT: [[E16]] = !{i64 8, !"_ZTSN5test31EE"}
// RV-OPT-LAYOUT: [[EF16]] = !{i64 8, !"_ZTSMN5test31EEFvvE.virtual"}

// MS: [[A8]] = !{i64 8, !"?AUA@@"}
// MS: [[B8]] = !{i64 8, !"?AUB@@"}
// MS: [[D8]] = !{i64 8, [[D_ID:![0-9]+]]}
// MS: [[D_ID]] = distinct !{}
// MS: [[FA8]] = !{i64 8, [[FA_ID:![0-9]+]]}
// MS: [[FA_ID]] = distinct !{}
