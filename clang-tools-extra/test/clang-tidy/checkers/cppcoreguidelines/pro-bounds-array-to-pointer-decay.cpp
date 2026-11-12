// RUN: %check_clang_tidy %s cppcoreguidelines-pro-bounds-array-to-pointer-decay %t
#include <stddef.h>

namespace gsl {
template <class T>
class array_view {
public:
  template <class U, size_t N>
  array_view(U (&arr)[N]);
};
}

void pointerfun(int *p);
void arrayfun(int p[]);
void arrayviewfun(gsl::array_view<int> &p);
size_t s();

void f() {
  int a[5];
  pointerfun(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not implicitly decay an array into a pointer; consider using gsl::array_view or an explicit cast instead [cppcoreguidelines-pro-bounds-array-to-pointer-decay]
  pointerfun((int *)a); // OK, explicit cast
  arrayfun(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not implicitly decay an array into a pointer

  pointerfun(a + s() - 10); // Convert to &a[g() - 10];
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not implicitly decay an array into a pointer

  gsl::array_view<int> av(a);
  arrayviewfun(av); // OK

  int i = a[0];      // OK
  int j = a[(1 + 2)];// OK
  pointerfun(&a[0]); // OK

  for (auto &e : a) // OK, iteration internally decays array to pointer
    e = 1;
}

const char *g() {
  return "clang"; // OK, decay string literal to pointer
}
const char *g2() {
    return ("clang"); // OK, ParenExpr hides the literal-pointer decay
}
const char *g3() {
    return __func__; // OK, don't diagnose PredefinedExpr
}

void f2(void *const *);
void bug25362() {
  void *a[2];
  f2(static_cast<void *const*>(a)); // OK, explicit cast
}

void issue31155(int i) {
  const char *a = i ? "foo" : "bar";    // OK, decay string literal to pointer
  const char *b = i ? "foo" : "foobar"; // OK, decay string literal to pointer

  char arr[1];
  const char *c = i ? arr : "bar";
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: do not implicitly decay an array into a pointer
  const char *d = i ? "foo" : arr;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: do not implicitly decay an array into a pointer
}
