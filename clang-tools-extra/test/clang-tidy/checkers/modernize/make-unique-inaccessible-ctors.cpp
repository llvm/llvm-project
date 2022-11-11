// RUN: %check_clang_tidy -std=c++14,c++17 -check-suffix=CXX-14-17 %s modernize-make-unique %t -- -- -I %S/Inputs/smart-ptr -D CXX_14_17=1
// RUN: %check_clang_tidy -std=c++20 -check-suffix=CXX-20 %s modernize-make-unique %t -- -- -I %S/Inputs/smart-ptr -D CXX_20=1

#include "unique_ptr.h"
// CHECK-FIXES: #include <memory>

struct NoCopyMoveCtor {
#ifdef CXX_20
  // C++20 requires to see the default constructor, otherwise it is illgal.
  NoCopyMoveCtor() = default;
#endif
#ifdef CXX_14_17
  int a, b;
#endif
  NoCopyMoveCtor(const NoCopyMoveCtor &) = delete; // implies move ctor is deleted
};

struct NoCopyMoveCtorVisible {
#ifdef CXX_20
  NoCopyMoveCtorVisible() = default;
#endif
private:
  NoCopyMoveCtorVisible(const NoCopyMoveCtorVisible&) = default;
  NoCopyMoveCtorVisible(NoCopyMoveCtorVisible&&) = default;
};

struct OnlyMoveCtor {
  OnlyMoveCtor() = default;
  OnlyMoveCtor(OnlyMoveCtor&&) = default;
  OnlyMoveCtor(const OnlyMoveCtor &) = delete;
};

struct OnlyCopyCtor {
#ifdef CXX_20
  OnlyCopyCtor() = default;
#endif
  OnlyCopyCtor(const OnlyCopyCtor&) = default;
  OnlyCopyCtor(OnlyCopyCtor&&) = delete;
};

struct OnlyCopyCtorVisible {
#ifdef CXX_20
  OnlyCopyCtorVisible() = default;
#endif
  OnlyCopyCtorVisible(const OnlyCopyCtorVisible &) = default;

private:
  OnlyCopyCtorVisible(OnlyCopyCtorVisible &&) = default;
};

struct ImplicitDeletedCopyCtor {
  const OnlyMoveCtor ctor;
};

void f() {
  auto my_ptr = std::unique_ptr<int>(new int(1));
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:17: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto my_ptr = std::make_unique<int>(1);
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:17: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto my_ptr = std::make_unique<int>(1);

  // "new NoCopyMoveCtor{}" is processed differently in C++14/17 and C++20:
  //   * In C++14/17, it is recognized as aggregate initialization,
  //     no fixes will be generated although the generated fix is compilable.
  //   * In C++20, it is is recognized as default constructor initialization
  //     (similar to "new NoCopyMoveCtor()"), the check will emit the fix and
  //     the fix is correct.
  auto PNoCopyMoveCtor = std::unique_ptr<NoCopyMoveCtor>(new NoCopyMoveCtor{});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:26: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto PNoCopyMoveCtor = std::unique_ptr<NoCopyMoveCtor>(new NoCopyMoveCtor{});
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:26: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto PNoCopyMoveCtor = std::make_unique<NoCopyMoveCtor>();

  auto PNoCopyMoveCtorVisible = std::unique_ptr<NoCopyMoveCtorVisible>(new NoCopyMoveCtorVisible{});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:33: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto PNoCopyMoveCtorVisible = std::unique_ptr<NoCopyMoveCtorVisible>(new NoCopyMoveCtorVisible{});
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:33: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto PNoCopyMoveCtorVisible = std::make_unique<NoCopyMoveCtorVisible>();

  auto POnlyMoveCtor = std::unique_ptr<OnlyMoveCtor>(new OnlyMoveCtor{});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:24: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto POnlyMoveCtor = std::unique_ptr<OnlyMoveCtor>(new OnlyMoveCtor{});
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:24: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto POnlyMoveCtor = std::make_unique<OnlyMoveCtor>();

  auto POnlyCopyCtor = std::unique_ptr<OnlyCopyCtor>(new OnlyCopyCtor{});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:24: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto POnlyCopyCtor = std::unique_ptr<OnlyCopyCtor>(new OnlyCopyCtor{});
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:24: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto POnlyCopyCtor = std::make_unique<OnlyCopyCtor>();

  auto POnlyCopyCtorVisible = std::unique_ptr<OnlyCopyCtorVisible>(new OnlyCopyCtorVisible{});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:31: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto POnlyCopyCtorVisible = std::unique_ptr<OnlyCopyCtorVisible>(new OnlyCopyCtorVisible{});
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:31: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto POnlyCopyCtorVisible = std::make_unique<OnlyCopyCtorVisible>();

  // This is aggregate initialization in C++20, no fix will be generated.
  auto PImplicitDeletedCopyCtor = std::unique_ptr<ImplicitDeletedCopyCtor>(new ImplicitDeletedCopyCtor{});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:35: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto PImplicitDeletedCopyCtor = std::unique_ptr<ImplicitDeletedCopyCtor>(new ImplicitDeletedCopyCtor{});
  // CHECK-MESSAGES-CXX-20: :[[@LINE-3]]:35: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-20: auto PImplicitDeletedCopyCtor = std::unique_ptr<ImplicitDeletedCopyCtor>(new ImplicitDeletedCopyCtor{});


#ifdef CXX_14_17
  // FIXME: it is impossible to use make_unique for this case, the check should
  // stop emitting the message.
  auto PNoCopyMoveCtor2 = std::unique_ptr<NoCopyMoveCtor>(new NoCopyMoveCtor{1, 2});
  // CHECK-MESSAGES-CXX-14-17: :[[@LINE-1]]:27: warning: use std::make_unique instead
  // CHECK-FIXES-CXX-14-17: auto PNoCopyMoveCtor2 = std::unique_ptr<NoCopyMoveCtor>(new NoCopyMoveCtor{1, 2});
#endif
}
