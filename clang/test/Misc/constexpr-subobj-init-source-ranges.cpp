// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info -fexperimental-new-constant-interpreter %s 2>&1 | FileCheck %s --strict-whitespace

struct DelBase {
  constexpr DelBase() = delete;
};

// CHECK:      :{[[@LINE+1]]:21-[[@LINE+1]]:28}
struct Foo : public DelBase {
  constexpr Foo() {};
};
constexpr Foo f;
