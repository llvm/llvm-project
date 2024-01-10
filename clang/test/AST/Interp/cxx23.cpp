// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=ref20 %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fcxx-exceptions -verify=ref23 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=expected20 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fcxx-exceptions -verify=expected23 %s -fexperimental-new-constant-interpreter


/// FIXME: The new interpreter is missing all the 'control flows through...' diagnostics.

constexpr int f(int n) {  // ref20-error {{constexpr function never produces a constant expression}} \
                          // ref23-error {{constexpr function never produces a constant expression}}
  static const int m = n; // ref20-note {{control flows through the definition of a static variable}} \
                          // ref20-warning {{is a C++23 extension}} \
                          // ref23-note {{control flows through the definition of a static variable}} \
                          // expected20-warning {{is a C++23 extension}}

  return m;
}
constexpr int g(int n) {        // ref20-error {{constexpr function never produces a constant expression}} \
                                // ref23-error {{constexpr function never produces a constant expression}}
  thread_local const int m = n; // ref20-note {{control flows through the definition of a thread_local variable}} \
                                // ref20-warning {{is a C++23 extension}} \
                                // ref23-note {{control flows through the definition of a thread_local variable}} \
                                // expected20-warning {{is a C++23 extension}}
  return m;
}

constexpr int c_thread_local(int n) { // ref20-error {{constexpr function never produces a constant expression}} \
                                      // ref23-error {{constexpr function never produces a constant expression}}
  static _Thread_local int m = 0;     // ref20-note {{control flows through the definition of a thread_local variable}} \
                                      // ref20-warning {{is a C++23 extension}} \
                                      // ref23-note {{control flows through the definition of a thread_local variable}} \
                                      // expected20-warning {{is a C++23 extension}}
  return m;
}


constexpr int gnu_thread_local(int n) { // ref20-error {{constexpr function never produces a constant expression}} \
                                        // ref23-error {{constexpr function never produces a constant expression}}
  static __thread int m = 0;            // ref20-note {{control flows through the definition of a thread_local variable}} \
                                        // ref20-warning {{is a C++23 extension}} \
                                        // ref23-note {{control flows through the definition of a thread_local variable}} \
                                        // expected20-warning {{is a C++23 extension}}
  return m;
}

constexpr int h(int n) {  // ref20-error {{constexpr function never produces a constant expression}} \
                          // ref23-error {{constexpr function never produces a constant expression}}
  static const int m = n; // ref20-note {{control flows through the definition of a static variable}} \
                          // ref20-warning {{is a C++23 extension}} \
                          // ref23-note {{control flows through the definition of a static variable}} \
                          // expected20-warning {{is a C++23 extension}}
  return &m - &m;
}

constexpr int i(int n) {        // ref20-error {{constexpr function never produces a constant expression}} \
                                // ref23-error {{constexpr function never produces a constant expression}}
  thread_local const int m = n; // ref20-note {{control flows through the definition of a thread_local variable}} \
                                // ref20-warning {{is a C++23 extension}} \
                                // ref23-note {{control flows through the definition of a thread_local variable}} \
                                // expected20-warning {{is a C++23 extension}}
  return &m - &m;
}

constexpr int j(int n) {
  if (!n)
    return 0;
  static const int m = n; // ref20-warning {{is a C++23 extension}} \
                          // expected20-warning {{is a C++23 extension}}
  return m;
}
constexpr int j0 = j(0);

constexpr int k(int n) {
  if (!n)
    return 0;
  thread_local const int m = n; // ref20-warning {{is a C++23 extension}} \
                                // expected20-warning {{is a C++23 extension}}

  return m;
}
constexpr int k0 = k(0);

namespace StaticLambdas {
  constexpr auto static_capture_constexpr() {
    char n = 'n';
    return [n] static { return n; }(); // expected23-error {{a static lambda cannot have any captures}} \
                                       // expected20-error {{a static lambda cannot have any captures}} \
                                       // expected20-warning {{are a C++23 extension}} \
                                       // expected20-warning {{is a C++23 extension}} \
                                       // ref23-error {{a static lambda cannot have any captures}} \
                                       // ref20-error {{a static lambda cannot have any captures}} \
                                       // ref20-warning {{are a C++23 extension}} \
                                       // ref20-warning {{is a C++23 extension}}
  }
  static_assert(static_capture_constexpr()); // expected23-error {{static assertion expression is not an integral constant expression}} \
                                             // expected20-error {{static assertion expression is not an integral constant expression}} \
                                             // ref23-error {{static assertion expression is not an integral constant expression}} \
                                             // ref20-error {{static assertion expression is not an integral constant expression}}

  constexpr auto capture_constexpr() {
    char n = 'n';
    return [n] { return n; }();
  }
  static_assert(capture_constexpr());
}
