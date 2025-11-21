// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

struct Flub {
  int a = 123;
  // CIR: @_ZN4FlubC1ERKS_(%arg0: !cir.ptr<!rec_Flub> loc({{.*}}), %arg1: !cir.ptr<!rec_Flub> loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Flub, copy, trivial true>>
  // CIR: @_ZN4FlubC2EOS_(%arg0: !cir.ptr<!rec_Flub> loc({{.*}}), %arg1: !cir.ptr<!rec_Flub> loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Flub, move, trivial true>
  // CIR: @_ZN4FlubaSERKS_(%arg0: !cir.ptr<!rec_Flub> loc({{.*}}), %arg1: !cir.ptr<!rec_Flub> loc({{.*}})) -> !cir.ptr<!rec_Flub> special_member<#cir.cxx_assign<!rec_Flub, copy, trivial true>>
  // CIR: @_ZN4FlubaSEOS_(%arg0: !cir.ptr<!rec_Flub> loc({{.*}}), %arg1: !cir.ptr<!rec_Flub> loc({{.*}})) -> !cir.ptr<!rec_Flub> special_member<#cir.cxx_assign<!rec_Flub, move, trivial true>>
};

struct Foo {
  int a;

  // CIR: @_ZN3FooC2Ev(%arg0: !cir.ptr<!rec_Foo> loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Foo, default>>
  Foo() : a(123) {}

  // CIR: @_ZN3FooC2ERKS_(%arg0: !cir.ptr<!rec_Foo> loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Foo, copy>>
  Foo(const Foo &other) : a(other.a) {}

  // CIR: @_ZN3FooC2EOS_(%arg0: !cir.ptr<!rec_Foo> loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> loc({{.*}})) special_member<#cir.cxx_ctor<!rec_Foo, move>>
  Foo(Foo &&other) noexcept : a(other.a) { other.a = 0; }

  // CIR: @_ZN3FooaSERKS_(%arg0: !cir.ptr<!rec_Foo> loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> loc({{.*}})) -> !cir.ptr<!rec_Foo> special_member<#cir.cxx_assign<!rec_Foo, copy>>
  Foo &operator=(const Foo &other) {
    if (this != &other) {
      a = other.a;
    }
    return *this;
  }

  // CIR: @_ZN3FooaSEOS_(%arg0: !cir.ptr<!rec_Foo> loc({{.*}}), %arg1: !cir.ptr<!rec_Foo> loc({{.*}})) -> !cir.ptr<!rec_Foo> special_member<#cir.cxx_assign<!rec_Foo, move>>
  Foo &operator=(Foo &&other) noexcept {
    if (this != &other) {
      a = other.a;
      other.a = 0;
    }
    return *this;
  }

  // CIR: @_ZN3FooD1Ev(!cir.ptr<!rec_Foo>) special_member<#cir.cxx_dtor<!rec_Foo>>
  ~Foo();
};

void trivial() {
  Flub f1{};
  Flub f2 = f1;
  Flub f3 = static_cast<Flub&&>(f1);
  f2 = f1;
  f1 = static_cast<Flub&&>(f3);
}

void non_trivial() {
  Foo f1{};
  Foo f2 = f1;
  Foo f3 = static_cast<Foo&&>(f1);
  f2 = f1;
  f1 = static_cast<Foo&&>(f3);
}
