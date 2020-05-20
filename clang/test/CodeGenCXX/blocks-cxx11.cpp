// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - | FileCheck %s

template <class T> void takeItByValue(T);
void takeABlock(void (^)());

// rdar://problem/11022704
namespace test_int {
  void test() {
    const int x = 100;
    takeABlock(^{ takeItByValue(x); });
    // CHECK: call void @_Z13takeItByValueIiEvT_(i32 100)
  }
}

namespace test_int_ref {
  void test() {
    const int y = 200;
    const int &x = y;
    takeABlock(^{ takeItByValue(x); });

    // TODO: there's no good reason that this isn't foldable.
    // CHECK: call void @_Z13takeItByValueIiEvT_(i32 {{%.*}})
  }
}

namespace test_float {
  void test() {
    const float x = 1;
    takeABlock(^{ takeItByValue(x); });
    // CHECK: call void @_Z13takeItByValueIfEvT_(float 1.0
  }
}

namespace test_float_ref {
  void test() {
    const float y = 100;
    const float &x = y;
    takeABlock(^{ takeItByValue(x); });

    // TODO: there's no good reason that this isn't foldable.
    // CHECK: call void @_Z13takeItByValueIfEvT_(float {{%.*}})
  }
}

namespace test_complex_int {
  void test() {
    constexpr _Complex int x = 500;
    takeABlock(^{ takeItByValue(x); });
    // CHECK:      store { i32, i32 } { i32 500, i32 0 },

    // CHECK:      store i32 500,
    // CHECK-NEXT: store i32 0,
    // CHECK-NEXT: [[COERCE:%.*]] = bitcast
    // CHECK-NEXT: [[CVAL:%.*]] = load i64, i64* [[COERCE]]
    // CHECK-NEXT: call void @_Z13takeItByValueICiEvT_(i64 [[CVAL]])
  }
}

namespace test_complex_int_ref {
  void test() {
    const _Complex int y = 100;
    const _Complex int &x = y;
    takeABlock(^{ takeItByValue(x); });
    // CHECK: call void @_Z13takeItByValueICiEvT_(i64
  }
}

namespace test_complex_int_ref_mutable {
  _Complex int y = 100;
  void test() {
    const _Complex int &x = y;
    takeABlock(^{ takeItByValue(x); });
    // CHECK:      [[R:%.*]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_ZN28test_complex_int_ref_mutable1yE, i32 0, i32 0)
    // CHECK-NEXT: [[I:%.*]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_ZN28test_complex_int_ref_mutable1yE, i32 0, i32 1)
    // CHECK-NEXT: [[RSLOT:%.*]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[CSLOT:%.*]], i32 0, i32 0
    // CHECK-NEXT: [[ISLOT:%.*]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[CSLOT]], i32 0, i32 1
    // CHECK-NEXT: store i32 [[R]], i32* [[RSLOT]]
    // CHECK-NEXT: store i32 [[I]], i32* [[ISLOT]]
    // CHECK-NEXT: [[COERCE:%.*]] = bitcast { i32, i32 }* [[CSLOT]] to i64*
    // CHECK-NEXT: [[CVAL:%.*]] = load i64, i64* [[COERCE]],
    // CHECK-NEXT: call void @_Z13takeItByValueICiEvT_(i64 [[CVAL]])
  }
}

// rdar://13295759
namespace test_block_in_lambda {
  void takeBlock(void (^block)());

  // The captured variable has to be non-POD so that we have a copy expression.
  struct A {
    void *p;
    A(const A &);
    ~A();
    void use() const;
  };

  void test(A a) {
    auto lambda = [a]() {
      takeBlock(^{ a.use(); });
    };
    lambda(); // make sure we emit the invocation function
  }
  // CHECK-LABEL:    define internal void @"_ZZN20test_block_in_lambda4testENS_1AEENK3$_0clEv"(
  // CHECK:      [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]], align 8
  // CHECK:      [[THIS:%.*]] = load [[LAMBDA_T:%.*]]*, [[LAMBDA_T:%.*]]**
  // CHECK:      [[TO_DESTROY:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]], [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[LAMBDA_T]], [[LAMBDA_T]]* [[THIS]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN20test_block_in_lambda1AC1ERKS0_({{.*}}* [[T0]], {{.*}}* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK-NEXT: call void @_ZN20test_block_in_lambda9takeBlockEU13block_pointerFvvE(void ()* [[T0]])
  // CHECK-NEXT: call void @_ZN20test_block_in_lambda1AD1Ev({{.*}}* [[TO_DESTROY]])
  // CHECK-NEXT: ret void
}
