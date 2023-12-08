// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 -std=c++11 | FileCheck %s

void bar() {
  // CHECK-DAG: @__FUNCTION__.___Z3barv_block_invoke = private unnamed_addr constant [17 x i8] c"bar_block_invoke\00", align 1
  const char * (^block1)() = ^() {
    return __FUNCTION__;
  };
  // CHECK-DAG: @__FUNCTION__.___Z3barv_block_invoke_2 = private unnamed_addr constant [19 x i8] c"bar_block_invoke_2\00", align 1
  const char * (^block2)() = ^() {
    return __FUNCTION__;
  };
}

void baz() {
  // CHECK-DAG: @__PRETTY_FUNCTION__.___Z3bazv_block_invoke = private unnamed_addr constant [24 x i8] c"void baz()_block_invoke\00", align 1
  const char * (^block1)() = ^() {
    return __PRETTY_FUNCTION__;
  };
  // CHECK-DAG: @__PRETTY_FUNCTION__.___Z3bazv_block_invoke_2 = private unnamed_addr constant [26 x i8] c"void baz()_block_invoke_2\00", align 1
  const char * (^block2)() = ^() {
    return __PRETTY_FUNCTION__;
  };
}

namespace foonamespace {
class Foo {
public:
  Foo() {
    // CHECK-DAG: @__PRETTY_FUNCTION__.___ZN12foonamespace3FooC2Ev_block_invoke = private unnamed_addr constant [38 x i8] c"foonamespace::Foo::Foo()_block_invoke\00", align 1
    const char * (^block1)() = ^() {
      return __PRETTY_FUNCTION__;
    };
    // CHECK-DAG: @__PRETTY_FUNCTION__.___ZN12foonamespace3FooC2Ev_block_invoke_2 = private unnamed_addr constant [40 x i8] c"foonamespace::Foo::Foo()_block_invoke_2\00", align 1
    const char * (^block2)() = ^() {
      return __PRETTY_FUNCTION__;
    };
    // CHECK-DAG: @__func__.___ZN12foonamespace3FooC2Ev_block_invoke_3 = private unnamed_addr constant [19 x i8] c"Foo_block_invoke_3\00", align 1
    const char * (^block3)() = ^() {
      return __func__;
    };
    bar();
    inside_lambda();
  }
  ~Foo() {
    // CHECK-DAG: @__func__.___ZN12foonamespace3FooD2Ev_block_invoke = private unnamed_addr constant [18 x i8] c"~Foo_block_invoke\00", align 1
    const char * (^block1)() = ^() {
      return __func__;
    };
    // CHECK-DAG: @__PRETTY_FUNCTION__.___ZN12foonamespace3FooD2Ev_block_invoke_2 = private unnamed_addr constant [41 x i8] c"foonamespace::Foo::~Foo()_block_invoke_2\00", align 1
    const char * (^block2)() = ^() {
      return __PRETTY_FUNCTION__;
    };
  }
  void bar() {
    // CHECK-DAG: @__PRETTY_FUNCTION__.___ZN12foonamespace3Foo3barEv_block_invoke = private unnamed_addr constant [43 x i8] c"void foonamespace::Foo::bar()_block_invoke\00", align 1
    const char * (^block1)() = ^() {
      return __PRETTY_FUNCTION__;
    };
    // CHECK-DAG: @__PRETTY_FUNCTION__.___ZN12foonamespace3Foo3barEv_block_invoke_2 = private unnamed_addr constant [45 x i8] c"void foonamespace::Foo::bar()_block_invoke_2\00", align 1
    const char * (^block2)() = ^() {
      return __PRETTY_FUNCTION__;
    };
    // CHECK-DAG: @__func__.___ZN12foonamespace3Foo3barEv_block_invoke_3 = private unnamed_addr constant [19 x i8] c"bar_block_invoke_3\00", align 1
    const char * (^block3)() = ^() {
      return __func__;
    };
  }
  void inside_lambda() {
    auto lambda = []() {
      // CHECK-DAG: @__PRETTY_FUNCTION__.___ZZN12foonamespace3Foo13inside_lambdaEvENKUlvE_clEv_block_invoke = private unnamed_addr constant [92 x i8] c"auto foonamespace::Foo::inside_lambda()::(anonymous class)::operator()() const_block_invoke\00", align 1
      const char * (^block1)() = ^() {
        return __PRETTY_FUNCTION__;
      };
      // CHECK-DAG: 	  @__PRETTY_FUNCTION__.___ZZN12foonamespace3Foo13inside_lambdaEvENKUlvE_clEv_block_invoke_2 = private unnamed_addr constant [94 x i8] c"auto foonamespace::Foo::inside_lambda()::(anonymous class)::operator()() const_block_invoke_2\00", align 1
      const char * (^block2)() = ^() {
        return __PRETTY_FUNCTION__;
      };
      // CHECK-DAG: @__func__.___ZZN12foonamespace3Foo13inside_lambdaEvENKUlvE_clEv_block_invoke_3 = private unnamed_addr constant [26 x i8] c"operator()_block_invoke_3\00", align 1
      const char * (^block3)() = ^() {
        return __func__;
      };
    };
    lambda();
  }
};
Foo f;
}
