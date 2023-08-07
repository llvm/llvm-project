// RUN: %clang_cc1 %s -fsyntax-only -verify=c
// RUN: %clang_cc1 -x c++ %s -fsyntax-only -verify=cxx

// cxx-no-diagnostics


/// Zero-sized structs should not crash.
int b() {
  struct {      } a[10];
  __builtin_memcpy(&a[2], a, 2); // c-warning {{buffer has size 0, but size argument is 2}}
  return 0;
}

#ifdef __cplusplus
// FIXME: This is UB and GCC correctly diagnoses it. Clang should do the same.
constexpr int b2() {
  struct {      } a[10];
  __builtin_memcpy(&a[2], a, 2);
  return 0;
}
static_assert(b2() == 0, "");
#endif
