// RUN: %check_clang_tidy -check-suffix=STRICT  %s cppcoreguidelines-pro-type-const-cast %t -- -config="{CheckOptions: {StrictMode: true}}"
// RUN: %check_clang_tidy -check-suffix=NSTRICT %s cppcoreguidelines-pro-type-const-cast %t

namespace Const {
const int *i;
int *j;

void f() {
  j = const_cast<int *>(i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:7: warning: do not use const_cast to remove const qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:7: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = const_cast<const int*>(j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:7: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  j = *const_cast<int **>(&i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:8: warning: do not use const_cast to remove const qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = *const_cast<const int**>(&j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  j = &const_cast<int&>(*i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:8: warning: do not use const_cast to remove const qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = &const_cast<const int&>(*j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]
}
}

namespace Volatile {
volatile int *i;
int *j;

void f() {
  j = const_cast<int *>(i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:7: warning: do not use const_cast to remove volatile qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:7: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = const_cast<volatile int*>(j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:7: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  j = *const_cast<int **>(&i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:8: warning: do not use const_cast to remove volatile qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = *const_cast<volatile int**>(&j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  j = &const_cast<int&>(*i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:8: warning: do not use const_cast to remove volatile qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = &const_cast<volatile int&>(*j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]
}
}

namespace ConstAndVolatile {
const volatile int *i;
int *j;

void f() {
  j = const_cast<int *>(i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:7: warning: do not use const_cast to remove const and volatile qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:7: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = const_cast<const volatile int*>(j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:7: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  j = *const_cast<int **>(&i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:8: warning: do not use const_cast to remove const and volatile qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = *const_cast<const volatile int**>(&j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  j = &const_cast<int&>(*i);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:8: warning: do not use const_cast to remove const and volatile qualifier [cppcoreguidelines-pro-type-const-cast]
  // CHECK-MESSAGES-STRICT:  :[[@LINE-2]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]

  i = &const_cast<const volatile int&>(*j);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:8: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]
}
}
