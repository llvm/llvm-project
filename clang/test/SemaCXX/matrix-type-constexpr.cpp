// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -std=c++11 -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -std=c++23 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -std=c++23 -verify %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

// GH232999

typedef float fx3x3_t __attribute__((matrix_type(3, 3)));
typedef int ix2x4_t __attribute__((matrix_type(2, 4)));

fx3x3_t GlobalFloat{};
ix2x4_t GlobalInt{};

#if __cplusplus >= 202302L
constexpr bool f() {
  fx3x3_t M{};
  ix2x4_t N{};
  fx3x3_t O = fx3x3_t();
  return true;
}
static_assert(f());
#endif
