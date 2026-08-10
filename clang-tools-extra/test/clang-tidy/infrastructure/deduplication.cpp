// RUN: %check_clang_tidy %s google-explicit-constructor %t

template<typename T>
struct A { A(T); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

void f() {
  A<int> a(0);
  A<double> b(0);
}
