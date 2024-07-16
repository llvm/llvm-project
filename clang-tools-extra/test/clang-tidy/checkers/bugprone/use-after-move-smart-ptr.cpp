// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-use-after-move %t -- \
// RUN:    -config="{CheckOptions: {bugprone-use-after-move.IgnoreNonDerefSmartPtrs: false}}" -- -fno-delayed-template-parsing -I %S/../modernize/Inputs/smart-ptr/

#include "unique_ptr.h"

namespace PR90174 {

struct A {};

struct SinkA {
  SinkA(std::unique_ptr<A>);
};

class ClassB {
  ClassB(std::unique_ptr<A> aaa) : aa(std::move(aaa)) {
    a = std::make_unique<SinkA>(std::move(aaa));
    // CHECK-MESSAGES: [[@LINE-1]]:43: warning: 'aaa' used after it was moved
    // CHECK-MESSAGES: [[@LINE-3]]:36: note: move occurred here
  }
  std::unique_ptr<A> aa;
  std::unique_ptr<SinkA> a;
};

void s(const std::unique_ptr<A> &);

template <typename T, typename... Args> auto my_make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

void natively(std::unique_ptr<A> x) {
  std::unique_ptr<A> tmp = std::move(x);
  std::unique_ptr<SinkA> y2{new SinkA(std::move(x))};
  // CHECK-MESSAGES: [[@LINE-1]]:49: warning: 'x' used after it was moved
  // CHECK-MESSAGES: [[@LINE-3]]:28: note: move occurred here
}

void viaStdMakeUnique(std::unique_ptr<A> x) {
  std::unique_ptr<A> tmp = std::move(x);
  std::unique_ptr<SinkA> y2 =
      std::make_unique<SinkA>(std::move(x));
  // CHECK-MESSAGES: [[@LINE-1]]:41: warning: 'x' used after it was moved
  // CHECK-MESSAGES: [[@LINE-4]]:28: note: move occurred here
}

void viaMyMakeUnique(std::unique_ptr<A> x) {
  std::unique_ptr<A> tmp = std::move(x);
  std::unique_ptr<SinkA> y2 = my_make_unique<SinkA>(std::move(x));
  // CHECK-MESSAGES: [[@LINE-1]]:63: warning: 'x' used after it was moved
  // CHECK-MESSAGES: [[@LINE-3]]:28: note: move occurred here
}

void viaMyMakeUnique2(std::unique_ptr<A> x) {
  std::unique_ptr<A> tmp = std::move(x);
  s(x);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: 'x' used after it was moved
  // CHECK-MESSAGES: [[@LINE-3]]:28: note: move occurred here
}

}
