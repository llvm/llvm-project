// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x hip %s
// RUN: not %clang_cc1 -fsyntax-only %s -DSLOC_CHECK 2>&1 | FileCheck %s

// Make sure operator followed by <<< is parsed as << and < since int CUDA/HIP
// it can never be a kernel launch expression.

template <typename T, typename T1> void operator<<(T, T1); // expected-error {{overloaded 'operator<<' must have at least one parameter of class or enumeration type}} \
                                                            // expected-note {{candidate template ignored: substitution failure [with T = int, T1 = int]}}

struct S1 {};

template <> void operator<<<>(S1, S1);

class C {
public:
  template <typename T> void operator<<(T) {}
};

void foobar() {
  C CC;
  CC.operator<<<int>(1);
  CC.template operator<<<int>(1);
#ifdef SLOC_CHECK
  // In CUDA/HIP mode <<< is a single token that gets split into << and <.
  // Verify that the < retains the correct source location after the split.
  CC.operator<<<int;
  // CHECK: [[@LINE-1]]:20: error: expected '>'
  // CHECK-NEXT: CC.operator<<<int;
  // CHECK-NEXT:                  ^
  // CHECK-NEXT: [[@LINE-4]]:16: note: to match this '<'
  // CHECK-NEXT: CC.operator<<<int;
  // CHECK-NEXT:              ^
  CC.template operator<<<int;
  // CHECK: [[@LINE-1]]:29: error: expected '>'
  // CHECK-NEXT: CC.template operator<<<int;
  // CHECK-NEXT:                           ^
  // CHECK-NEXT: [[@LINE-4]]:25: note: to match this '<'
  // CHECK-NEXT: CC.template operator<<<int;
  // CHECK-NEXT:                       ^
#endif
}

// Verify TryParseOperatorId handles tok::lesslessless as well, so the invalid
// code below produces clearer errors.
template<typename T> int operator<<(int, T) { return 0; } // expected-error {{overloaded 'operator<<' must have at least one parameter of class or enumeration type}} \
                                                            // expected-note {{candidate template ignored: substitution failure [with T = int]}}

void test() {
  int(operator<<<int>(1, 2)); // expected-error {{no matching function for call to 'operator<<'}} \
                               // expected-note {{in instantiation of function template specialization 'operator<<<int>' requested here}} \
                               // expected-note {{in instantiation of function template specialization 'operator<<<int, int>' requested here}}
}
