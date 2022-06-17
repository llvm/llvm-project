// RUN: %check_clang_tidy -std=c++14-or-later %s cppcoreguidelines-pro-bounds-pointer-arithmetic %t

// Fix PR36489 and detect auto-deduced value correctly.
char *getPtr();
auto getPtrAuto() { return getPtr(); }
decltype(getPtr()) getPtrDeclType();
decltype(auto) getPtrDeclTypeAuto() { return getPtr(); }
auto getPtrWithTrailingReturnType() -> char *;

void auto_deduction_binary() {
  auto p1 = getPtr() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not use pointer arithmetic
  auto p2 = getPtrAuto() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not use pointer arithmetic
  auto p3 = getPtrWithTrailingReturnType() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: do not use pointer arithmetic
  auto p4 = getPtr();
  auto *p5 = getPtr();
  p4 = p4 + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not use pointer arithmetic
  p5 = p5 + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not use pointer arithmetic
  auto p6 = getPtrDeclType() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: do not use pointer arithmetic
  auto p7 = getPtrDeclTypeAuto() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: do not use pointer arithmetic
  auto *p8 = getPtrDeclType() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: do not use pointer arithmetic
  auto *p9 = getPtrDeclTypeAuto() + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: do not use pointer arithmetic
}

void auto_deduction_subscript() {
  char p1 = getPtr()[2];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic
  auto p2 = getPtr()[3];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic

  char p3 = getPtrAuto()[4];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic
  auto p4 = getPtrAuto()[5];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic

  char p5 = getPtrWithTrailingReturnType()[6];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic
  auto p6 = getPtrWithTrailingReturnType()[7];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic

  auto p7 = getPtrDeclType()[8];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic
  auto p8 = getPtrDeclTypeAuto()[9];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use pointer arithmetic
}
