// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only %s
// expected-no-diagnostics

int main() {
  double x = 37;

  __auto_type _Atomic xa = x;
  _Atomic __auto_type ax = x;
  
  _Static_assert(
      __builtin_types_compatible_p(__typeof(xa), _Atomic double),
      "incorrect xa type");

  _Static_assert(
      __builtin_types_compatible_p(__typeof(ax), _Atomic double),
      "incorrect ax type");

  _Static_assert(
      __builtin_types_compatible_p(_Atomic double, __typeof(xa)),
      "incorrect");

  _Static_assert(
      __builtin_types_compatible_p(_Atomic double, __typeof(ax)),
      "incorrect");
  return 0;
}
