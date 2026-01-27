// RUN: %check_clang_tidy %s misc-static-initialization-cycle %t -- -- -fno-delayed-template-parsing

namespace simple_cycle {
struct S { static int A; };

int B = S::A;
int S::A = B;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: Static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-4]]:9: note: Value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: Value of 'B' may be used to initialize variable 'A' here

namespace self_init {
struct S { static int A; };
int S::A = S::A;
}
// CHECK-NOTES: :[[@LINE-3]]:23: warning: Static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-3]]:12: note: Value of 'A' may be used to initialize variable 'A' here

namespace cycle_at_end {
struct S { static int A; };

int B = 1;
int C = B + S::A;
int S::A = C;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: Static variable initialization cycle detected involving 'C'
// CHECK-NOTES: :[[@LINE-4]]:13: note: Value of 'A' may be used to initialize variable 'C' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: Value of 'C' may be used to initialize variable 'A' here

namespace cycle_at_start {
struct S { static int A; };

int B = S::A;
int S::A = B;
int C = B + 1;
}
// CHECK-NOTES: :[[@LINE-4]]:5: warning: Static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-5]]:9: note: Value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-5]]:12: note: Value of 'B' may be used to initialize variable 'A' here

namespace multiple_cycle {
struct S { static int A; };

int B = S::A;
int C = S::A;
int S::A = B + C;
}
// CHECK-NOTES: :[[@LINE-3]]:5: warning: Static variable initialization cycle detected involving 'C'
// CHECK-NOTES: :[[@LINE-4]]:9: note: Value of 'A' may be used to initialize variable 'C' here
// CHECK-NOTES: :[[@LINE-4]]:16: note: Value of 'C' may be used to initialize variable 'A' here

namespace long_cycle {
struct S { static int A; };

int B = S::A;
int C = B + 1;
int S::A = C;
}
// CHECK-NOTES: :[[@LINE-4]]:5: warning: Static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-5]]:9: note: Value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:12: note: Value of 'C' may be used to initialize variable 'A' here
// CHECK-NOTES: :[[@LINE-6]]:9: note: Value of 'B' may be used to initialize variable 'C' here

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
// CHECK-NOTES: :[[@LINE-3]]:5: warning: Static variable initialization cycle detected involving 'B'
// CHECK-NOTES: :[[@LINE-4]]:9: note: Value of 'A' may be used to initialize variable 'B' here
// CHECK-NOTES: :[[@LINE-4]]:15: note: Value of 'B' may be used to initialize variable 'A' here

namespace func_static_ref_1 {
struct S { static int A; };
int f1() {
  return S::A;
}
int S::A = f1();
}
// CHECK-NOTES: :[[@LINE-6]]:23: warning: Static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-5]]:10: note: Value of 'A' may be used to compute result of 'f1'
// CHECK-NOTES: :[[@LINE-4]]:12: note: Result of 'f1' may be used to initialize variable 'A' here

namespace func_static_ref_2 {
struct S { static int A; };
int f1() {
  static int X = S::A;
  return 1;
}
int S::A = f1();
}
// CHECK-NOTES: :[[@LINE-7]]:23: warning: Static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-6]]:18: note: Value of 'A' may be used to compute result of 'f1'
// CHECK-NOTES: :[[@LINE-4]]:12: note: Result of 'f1' may be used to initialize variable 'A' here

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
  return sizeof(A);
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
// CHECK-NOTES: :[[@LINE-6]]:14: warning: Static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-7]]:18: note: Result of 'f' may be used to initialize variable 'A' here
// CHECK-NOTES: :[[@LINE-5]]:10: note: Value of 'A' may be used to compute result of 'f'

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
// CHECK-NOTES: :[[@LINE-11]]:12: warning: Static variable initialization cycle detected involving 'A'
// CHECK-NOTES: :[[@LINE-6]]:10: note: Value of 'A' may be used to compute result of 'f1'
// CHECK-NOTES: :[[@LINE-10]]:13: note: Result of 'f1' may be used to initialize variable 'A' here
