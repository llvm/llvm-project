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
    // CHECK:      store i32 500,

    // CHECK:      store i32 500,
    // CHECK-NEXT: store i32 0,
    // CHECK-NEXT: [[COERCE:%.*]] = bitcast
    // CHECK-NEXT: [[CVAL:%.*]] = load i64* [[COERCE]]
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
    // CHECK:      [[R:%.*]] = load i32* getelementptr inbounds ({ i32, i32 }* @_ZN28test_complex_int_ref_mutable1yE, i32 0, i32 0)
    // CHECK-NEXT: [[I:%.*]] = load i32* getelementptr inbounds ({ i32, i32 }* @_ZN28test_complex_int_ref_mutable1yE, i32 0, i32 1)
    // CHECK-NEXT: [[RSLOT:%.*]] = getelementptr inbounds { i32, i32 }* [[CSLOT:%.*]], i32 0, i32 0
    // CHECK-NEXT: [[ISLOT:%.*]] = getelementptr inbounds { i32, i32 }* [[CSLOT]], i32 0, i32 1
    // CHECK-NEXT: store i32 [[R]], i32* [[RSLOT]]
    // CHECK-NEXT: store i32 [[I]], i32* [[ISLOT]]
    // CHECK-NEXT: [[COERCE:%.*]] = bitcast { i32, i32 }* [[CSLOT]] to i64*
    // CHECK-NEXT: [[CVAL:%.*]] = load i64* [[COERCE]],
    // CHECK-NEXT: call void @_Z13takeItByValueICiEvT_(i64 [[CVAL]])
  }
}

