// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits \
// RUN:            -fsyntax-only %s 2>&1 | FileCheck %s

// TODO test we don't mess up vertical whitespace
// TODO test different whitespaces
// TODO test different contexts
  // when it's on the right side

void basic() {
  int *ptr;
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> ptr"
  *(ptr+5)=1;
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:5}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:8-[[@LINE-2]]:9}:"["
// CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:11}:"]"
}

// The weird preceding semicolon ensures that we preserve that range intact.
void char_ranges() {
  int *p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:9}:"std::span<int> p"

  ;* ( p + 5 ) = 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:8}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:9-[[@LINE-2]]:12}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:15}:"]"

  ;*   (p+5)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"]"

  ;*(   p+5)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"]"

  ;*(   p+5)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"]"

  ;*( p   +5)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:7}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:8-[[@LINE-2]]:12}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:14}:"]"

  ;*(p+   5)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"]"

  ;*(p+ 5   )= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:9}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:14}:"]"

  ;*(p+ 5)   = 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:9}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:11}:"]"

  ;   *(p+5)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:13}:"]"

  ;*(p+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:8}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:14-[[@LINE-3]]:15}:"]"

  ;*   (p+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:"]"

  ;*(   p+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:"]"

  ;*(   p+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:"]"

  ;*(p   +123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:"]"

  ;*(p+   123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:11}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:"]"

  ;*(p+123456   )= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:8}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:14-[[@LINE-3]]:18}:"]"

  ;*(p+123456)   = 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:8}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:14-[[@LINE-3]]:15}:"]"

  int *ptrrrrrr;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:16}:"std::span<int> ptrrrrrr"

  ;* ( ptrrrrrr + 123456 )= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:8}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:19}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:25-[[@LINE-3]]:27}:"]"

  ;*   (ptrrrrrr+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:18}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:25}:"]"

  ;*(   ptrrrrrr+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:18}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:25}:"]"

  ;*(   ptrrrrrr+123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:9}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:18}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:25}:"]"

  ;*(ptrrrrrr   +123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:18}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:25}:"]"

  ;*(ptrrrrrr+   123456)= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:18}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:25}:"]"

  ;*(ptrrrrrr+123456   )= 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:15}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:25}:"]"

  ;*(ptrrrrrr+123456)   = 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:15}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:22}:"]"
}

void base_on_rhs() {
  int* ptr;
  *(10 + ptr) = 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:5}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:10}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:14}:"]"
}

void many_parens() {
  int* ptr;
  *(( (10 + ptr)) ) = 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:13}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:20}:"]"
}

void lvaue_to_rvalue() {
  int * ptr;
  int tmp = *(ptr + 10);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:15}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:21}:"["
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:24}:"]"
}

// Fixits emitted for the cases below would be incorrect.
// CHECK-NOT: fix-it:
// Array subsctipt opertor of std::span accepts unsigned integer.
void negative() {
  int* ptr;
  *(ptr + -5) = 1; // skip
}

void subtraction() {
  int* ptr;
  *(ptr - 5) = 1; // skip
}

void subtraction_of_negative() {
  int* ptr;
  *(ptr - -5) = 1; // FIXME: implement fixit (uncommon case - low priority)
}


void bindingDecl(int *p, int *q) {
  int * a[2] = {p, q};
  auto [x, y] = a;

  *(x + 1) = 1; // FIXME: deal with `BindingDecl`s
}
