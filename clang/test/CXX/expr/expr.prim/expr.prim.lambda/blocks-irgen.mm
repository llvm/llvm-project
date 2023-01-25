// RUN: %clang_cc1 -std=c++11 -fblocks -emit-llvm -o - -triple x86_64-apple-darwin11.3 %s | FileCheck %s

namespace PR12746 {
  // CHECK: define{{.*}} zeroext i1 @_ZN7PR127462f1EPi
  bool f1(int *x) {
    // CHECK: store ptr @___ZN7PR127462f1EPi_block_invoke
    bool (^outer)() = ^ {
      auto inner = [&]() -> bool {
	return x == 0;
      };
      return inner();
    };
    return outer();
  }

  // CHECK: define internal noundef zeroext i1 @___ZN7PR127462f1EPi_block_invoke
  // CHECK: call noundef zeroext i1 @"_ZZZN7PR127462f1EPiEUb_ENK3$_0clEv"

  bool f2(int *x) {
    auto outer = [&]() -> bool {
      bool (^inner)() = ^ {
	return x == 0;
      };
      return inner();
    };
    return outer();
  }
}

