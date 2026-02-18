// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s -std=c++11\
// RUN:  -analyzer-output=text

#include "Inputs/system-header-simulator-cxx.h"

class C {
  int n;
public:
  void memberFun();
};

template <class... Ts> void opaqueFreeFun(Ts...);

class Foo {
  std::unique_ptr<C> up;
  void opaqueMemberFun();

  void testOpaqueMemberFun() {
    auto tmp = std::move(up);
    opaqueMemberFun(); // clears region state
    (void)up->memberFun(); // no-warning
  }

  void testOpaqueFreeFun() {
    auto tmp = std::move(up); // expected-note{{Smart pointer 'up' of type 'std::unique_ptr' is reset to null when moved from}}
    opaqueFreeFun(); // does not clear region state
    (void)up->memberFun(); // expected-warning{{Dereference of null smart pointer 'up' of type 'std::unique_ptr'}}
                           // expected-note@-1{{Dereference of null smart pointer 'up' of type 'std::unique_ptr'}}
  }
};

void testInstanceMemberFun(C c) {
  auto tmp1 = std::move(c); // expected-note{{Object 'c' is moved}}
  auto tmp2 = std::move(c); // expected-warning{{Moved-from object 'c' is moved}}
                            // expected-note@-1{{Moved-from object 'c' is moved}}
  auto tmp3 = std::move(c);
  c.memberFun(); // does not clear region state
  auto tmp4 = std::move(c);
  auto tmp5 = std::move(c); // no-warning
}

void testPassByRef(C c) {
  auto tmp1 = std::move(c); // expected-note{{Object 'c' is moved}}
  auto tmp2 = std::move(c); // expected-warning{{Moved-from object 'c' is moved}}
                            // expected-note@-1{{Moved-from object 'c' is moved}}
  auto tmp3 = std::move(c);
  opaqueFreeFun<C&>(c); // clears region state
  auto tmp4 = std::move(c); // expected-note{{Object 'c' is moved}}
  auto tmp5 = std::move(c); // expected-warning{{Moved-from object 'c' is moved}}
                            // expected-note@-1{{Moved-from object 'c' is moved}}
}
