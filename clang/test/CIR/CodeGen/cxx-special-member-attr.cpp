// RUN: split-file %s %t

//--- special_member_attr.cpp

// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -fclangir -emit-cir %t/special_member_attr.cpp -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %t/special_member_attr.cpp

struct Flub {
  int a = 123;
};

struct Foo {
  int a;

  Foo() : a(123) {}
  Foo(const Foo &other) : a(other.a) {}
  Foo(Foo &&other) noexcept : a(other.a) { other.a = 0; }

  Foo &operator=(const Foo &other) {
    if (this != &other) {
      a = other.a;
    }
    return *this;
  }

  Foo &operator=(Foo &&other) noexcept {
    if (this != &other) {
      a = other.a;
      other.a = 0;
    }
    return *this;
  }

  ~Foo();
};

// The trivial copy/move assignment operators are emitted at module scope with
// the special_member attribute, and their bodies perform a whole-object copy.
// CIR-LABEL: cir.func{{.*}} @_ZN4FlubaSERKS_(
// CIR-SAME: special_member<#cir.cxx_assign<!rec_Flub, copy, trivial true>>
// CIR: cir.copy {{.*}} : !cir.ptr<!rec_Flub>
// CIR-LABEL: cir.func{{.*}} @_ZN4FlubaSEOS_(
// CIR-SAME: special_member<#cir.cxx_assign<!rec_Flub, move, trivial true>>
// CIR: cir.copy {{.*}} : !cir.ptr<!rec_Flub>

void trivial_func() {
  Flub f1{};

  Flub f2 = f1;
  Flub f3 = static_cast<Flub &&>(f1);

  f2 = f1;
  f1 = static_cast<Flub &&>(f3);
}

// Trivial assignment keeps the operator= call; its body (above) does the copy.
// CIR-LABEL: cir.func{{.*}} @_Z12trivial_funcv(
// CIR: cir.call{{.*}}@_ZN4FlubaSERKS_
// CIR: cir.call{{.*}}@_ZN4FlubaSEOS_

void non_trivial_func() {
  Foo f1{};
  Foo f2 = f1;
  Foo f3 = static_cast<Foo &&>(f1);
  f2 = f1;
  f1 = static_cast<Foo &&>(f3);
}

// CIR-LABEL: cir.func{{.*}} @_ZN3FooC2Ev(
// CIR-SAME: special_member<#cir.cxx_ctor<!rec_Foo, default>>
// CIR-LABEL: cir.func{{.*}} @_ZN3FooC2ERKS_(
// CIR-SAME: special_member<#cir.cxx_ctor<!rec_Foo, copy>>
// CIR-LABEL: cir.func{{.*}} @_ZN3FooC2EOS_(
// CIR-SAME: special_member<#cir.cxx_ctor<!rec_Foo, move>>
// CIR-LABEL: cir.func{{.*}} @_ZN3FooaSERKS_(
// CIR-SAME: special_member<#cir.cxx_assign<!rec_Foo, copy>>
// CIR-LABEL: cir.func{{.*}} @_ZN3FooaSEOS_(
// CIR-SAME: special_member<#cir.cxx_assign<!rec_Foo, move>>
// CIR-LABEL: cir.func{{.*}} @_ZN3FooD1Ev(
// CIR-SAME: special_member<#cir.cxx_dtor<!rec_Foo>>

//--- trivial_union_assign.cpp

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/trivial_union_assign.cpp -o %t-union.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-union.cir %t/trivial_union_assign.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/trivial_union_assign.cpp -o %t-union-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-union-cir.ll %t/trivial_union_assign.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/trivial_union_assign.cpp -o %t-union.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-union.ll %t/trivial_union_assign.cpp

union YYSTYPE {
  void *yt_casestring;
  void *yt_ID;
};

extern YYSTYPE yylval;

static int consume(YYSTYPE v) { return v.yt_casestring != nullptr; }

int test_shift(YYSTYPE *yyvsp) {
  yylval.yt_casestring = reinterpret_cast<void *>(0x42);
  *++yyvsp = yylval;
  return consume(yyvsp[0]);
}

// The defaulted union copy-assignment operator copies the whole object in its
// body -- previously the body was a no-op that LLVM deleted, dropping the
// store at -O3 -- and the call is kept at the assignment site.
// CIR-LABEL: cir.func{{.*}} @_ZN7YYSTYPEaSERKS_(
// CIR-SAME: special_member<#cir.cxx_assign<!rec_YYSTYPE, copy, trivial true>>
// CIR: cir.copy {{.*}} : !cir.ptr<!rec_YYSTYPE>
// CIR-LABEL: cir.func{{.*}} @_Z10test_shiftP7YYSTYPE(
// CIR: cir.call{{.*}}@_ZN7YYSTYPEaSERKS_

// LLVM: define{{.*}} @_ZN7YYSTYPEaSERKS_(
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)
// LLVM-LABEL: define{{.*}} @_Z10test_shiftP7YYSTYPE(
// LLVM: call{{.*}}@_ZN7YYSTYPEaSERKS_

// Classic CodeGen inlines the trivial union assignment at the call site and
// emits no operator= function; the store is performed directly.
// OGCG-LABEL: define{{.*}} @_Z10test_shiftP7YYSTYPE(
// OGCG: store ptr inttoptr (i64 66 to ptr), ptr @yylval
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}@yylval, i64 8, i1 false)
// OGCG-NOT: @_ZN7YYSTYPEaSERKS_
