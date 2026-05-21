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

void trivial_func() {
  Flub f1{};

  Flub f2 = f1;
  // Trivial copy/move constructors are inlined as cir.copy
  // CIR: cir.copy {{.*}} : !cir.ptr<!rec_Flub>

  Flub f3 = static_cast<Flub&&>(f1);
  // CIR: cir.copy {{.*}} : !cir.ptr<!rec_Flub>

  f2 = f1;
  // CIR: cir.copy {{.*}} : !cir.ptr<!rec_Flub>
  // CIR-NOT: cir.call{{.*}}@_ZN4FlubaSERKS_
  f1 = static_cast<Flub&&>(f3);
  // CIR: cir.copy {{.*}} : !cir.ptr<!rec_Flub>
  // CIR-NOT: cir.call{{.*}}@_ZN4FlubaSEOS_
}

void non_trivial_func() {
  Foo f1{};
  // CIR: @_ZN3FooC2Ev(%arg0: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Foo, default>>

  Foo f2 = f1;
  // CIR: @_ZN3FooC2ERKS_(%arg0: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Foo, copy>>

  Foo f3 = static_cast<Foo&&>(f1);
  // CIR: @_ZN3FooC2EOS_(%arg0: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Foo, move>>

  f2 = f1;
  // CIR: @_ZN3FooaSERKS_(%arg0: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}})) -> (!cir.ptr<!rec_Foo>{{.*}}) special_member<#cir.cxx_assign<!rec_Foo, copy>>

  f1 = static_cast<Foo&&>(f3);
  // CIR: @_ZN3FooaSEOS_(%arg0: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> {{[{][^}]*[}]}} loc({{.*}})) -> (!cir.ptr<!rec_Foo>{{.*}}) special_member<#cir.cxx_assign<!rec_Foo, move>>
  // CIR: @_ZN3FooD1Ev(!cir.ptr<!rec_Foo> {{.*}}) special_member<#cir.cxx_dtor<!rec_Foo>>
}

//--- trivial_union_assign.cpp

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -fclangir -emit-cir %t/trivial_union_assign.cpp -o %t-union.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-union.cir %t/trivial_union_assign.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -fclangir -emit-llvm %t/trivial_union_assign.cpp -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %t/trivial_union_assign.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -emit-llvm %t/trivial_union_assign.cpp -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %t/trivial_union_assign.cpp

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

// CIR-LABEL: cir.func{{.*}} @_Z10test_shiftP7YYSTYPE
// CIR-NOT: cir.call{{.*}}@_ZN7YYSTYPEaSERKS_
// CIR: cir.copy

// LLVM-LABEL: define{{.*}} @_Z10test_shiftP7YYSTYPE(
// LLVM: store{{.*}}@yylval
// LLVM: store i64 66
// LLVM-NOT: readonly
// LLVM: ret i32 1
