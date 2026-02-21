// RUN: %check_clang_tidy %s misc-static-initialization-cycle %t -- -- -fno-delayed-template-parsing

namespace simple_cycle {
struct S { static int A; };

int B = S::A;
int S::A = B;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-4]]:9: note: value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: value of 'B' may be used to initialize variable 'A' here

namespace self_init {
struct S { static int A; };
int S::A = S::A;
}
// CHECK-NOTES: :[[@LINE-3]]:23: warning: static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-3]]:12: note: value of 'A' may be used to initialize variable 'A' here

namespace cycle_at_end {
struct S { static int A; };

int B = 1;
int C = B + S::A;
int S::A = C;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: static variable initialization cycle detected involving 'C'
// CHECK-NOTES: :[[@LINE-4]]:13: note: value of 'A' may be used to initialize variable 'C' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: value of 'C' may be used to initialize variable 'A' here

namespace cycle_at_start {
struct S { static int A; };

int B = S::A;
int S::A = B;
int C = B + 1;
}
// CHECK-NOTES: :[[@LINE-4]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-5]]:9: note: value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-5]]:12: note: value of 'B' may be used to initialize variable 'A' here

namespace multiple_cycle {
struct S { static int A; };

int B = S::A;
int C = S::A;
int S::A = B + C;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: static variable initialization cycle detected involving 'C'
// CHECK-NOTES: :[[@LINE-4]]:9: note: value of 'A' may be used to initialize variable 'C' here
// CHECK-NOTES: :[[@LINE-4]]:16: note: value of 'C' may be used to initialize variable 'A' here

namespace long_cycle {
struct S { static int A; };

int B = S::A;
int C = B + 1;
int S::A = C;
}
// CHECK-NOTES: :[[@LINE-4]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-5]]:9: note: value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: value of 'C' may be used to initialize variable 'A' here
// CHECK-NOTES: :[[@LINE-6]]:9: note: value of 'B' may be used to initialize variable 'C' here

namespace no_cycle {
int A = 2;
int B = A;
int C = B + A;
}

namespace init_expr {
struct S { static int A; };
int f1(int X, int Y);

int B = S::A + 1;
int S::A = f1(B, 2);
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-4]]:9: note: value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:15: note: value of 'B' may be used to initialize variable 'A' here

namespace func_static_ref_1 {
struct S { static int A; };
int f1() {
  return S::A;
}
int S::A = f1();
}
// CHECK-NOTES: :[[@LINE-6]]:23: warning: static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-5]]:10: note: value of 'A' may be used to compute result of 'f1'
// CHECK-NOTES: :[[@LINE-4]]:12: note: result of 'f1' may be used to initialize variable 'A' here

namespace func_static_ref_2 {
struct S { static int A; };
int f1() {
  static int X = S::A;
  return 1;
}
int S::A = f1();
}
// CHECK-NOTES: :[[@LINE-7]]:23: warning: static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-6]]:18: note: value of 'A' may be used to compute result of 'f1'
// CHECK-NOTES: :[[@LINE-4]]:12: note: result of 'f1' may be used to initialize variable 'A' here

namespace func_static_ref_3 {
struct S { static int A; };
int f1() {
  S::A = 3;
  return 34;
}
int S::A = f1();
}

namespace recursive_calls {
int f2();
int f1() {
  return f2();
}
int f2() {
  return f1();
}
int A = f1();
}

namespace use_static_compile_time {
int f() {
  static int A = f();
  static decltype(A) B = 2;
  return sizeof(A) + B;
}
}

namespace static_var_recursive_init {
int f(int i) {
  static int A = f(1);
  if (i == 1)
    return 1;
  return A + i;
}
}
// CHECK-NOTES: :[[@LINE-6]]:14: warning: static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-7]]:18: note: result of 'f' may be used to initialize variable 'A' here
// CHECK-NOTES: :[[@LINE-5]]:10: note: value of 'A' may be used to compute result of 'f'

namespace singleton {
struct S { int X; };

S *get_S() {
  static S *TheS;
  if (!TheS) {
    TheS = new S;
  }
  return TheS;
}
}

namespace compound_assign_in_func {
struct S { static int A; };
int f() {
  int local = 0;
  local += S::A;
  return local;
}
int S::A = f();
}
// CHECK-NOTES: :[[@LINE-8]]:23: warning: static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-6]]:12: note: value of 'A' may be used to compute result of 'f'
// CHECK-NOTES: :[[@LINE-4]]:12: note: result of 'f' may be used to initialize variable 'A' here

namespace compound_assign_lhs {
struct S { static int A; };
int f();
int B = S::A + f();
int f() {
  S::A -= B;
  return 1;
}
}
// CHECK-NOTES: :[[@LINE-6]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-5]]:11: note: value of 'B' may be used to compute result of 'f'
// CHECK-NOTES: :[[@LINE-8]]:16: note: result of 'f' may be used to initialize variable 'B' here

namespace template_test {
template <class T>
struct S {
  static T f1();
  static T A;
};
template <class T>
T S<T>::A = f1();
template <class T>
T S<T>::f1() {
  return A;
}

S<int> X;
}
// CHECK-NOTES: :[[@LINE-11]]:12: warning: static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-6]]:10: note: value of 'A' may be used to compute result of 'f1'
// CHECK-NOTES: :[[@LINE-10]]:13: note: result of 'f1' may be used to initialize variable 'A' here

namespace test_lambda_1 {
struct S { static int A; };
int B = []() { return S::A; }();
int S::A = B;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-4]]:23: note: value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: value of 'B' may be used to initialize variable 'A' here

namespace test_lambda_2 {
struct S { static int A; };
auto B = []() { return S::A; };
int S::A = B();
}
// this is not found by the check
// value of 'A' is not needed to initialize 'B'
// the check does not maintain values of variables (to find the stored
// lambda and relation to 'A')

namespace test_lambda_3 {
struct S { static int A; };
int f() {
  return []() { return 2 * S::A; }() + 3;
}
int B = f();
int S::A = B;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-6]]:28: note: value of 'A' may be used to compute result of 'f'
// CHECK-NOTES: :[[@LINE-4]]:12: note: value of 'B' may be used to initialize variable 'A' here
// CHECK-NOTES: :[[@LINE-6]]:9: note: result of 'f' may be used to initialize variable 'B' here

namespace test_lambda_4 {
template <class L>
void f1(L) {};

struct S { static int A; };

int f() {
  f1([]() { return 2 * S::A; });
  return 1;
}
int B = f();
int S::A = B;
}

namespace mixed_cycle {
struct S { static int A; };
int B = S::A;
int f_b() { return B + 1; }
int C = f_b();
int D = C + 1;
int S::A = []() { return D + 1; }();
}
// CHECK-NOTES: :[[@LINE-6]]:5: warning: static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-7]]:9: note: value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:26: note: value of 'D' may be used to initialize variable 'A' here
// CHECK-NOTES: :[[@LINE-6]]:9: note: value of 'C' may be used to initialize variable 'D' here
// CHECK-NOTES: :[[@LINE-8]]:9: note: result of 'f_b' may be used to initialize variable 'C' here
// CHECK-NOTES: :[[@LINE-10]]:20: note: value of 'B' may be used to compute result of 'f_b'
