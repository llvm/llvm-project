// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fsafe-buffer-usage-suggestions -include %s %s 2>&1 | FileCheck %s

// We cannot deal with overload conflicts for now so NO fix-it to
// function parameters will be emitted if there are overloads for that
// function.

#ifndef INCLUDE_ME
#define INCLUDE_ME

void baz();

#else


void foo(int *p, int * q);

void foo(int *p);

void foo(int *p) {
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int tmp;
  tmp = p[5];
}

// an overload declaration of `bar(int *)` appears after it
void bar(int *p) {
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int tmp;
  tmp = p[5];
}

void bar();

// an overload declaration of `baz(int)` appears is included
void baz(int *p) {
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int tmp;
  tmp = p[5];
}

namespace NS {
  // `NS::foo` is a distinct function from `foo`, so it has no
  // overload and shall be fixed.
  void foo(int *p) {
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:12-[[@LINE-1]]:18}:"std::span<int> p"
    int tmp;
    tmp = p[5];
  }
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:4}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void foo(int *p) {return foo(std::span<int>(p, <# size #>));}\n"

  // Similarly, `NS::bar` is distinct from `bar`:
  void bar(int *p);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:3}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:19-[[@LINE-2]]:19}:";\nvoid bar(std::span<int> p)"
} // end of namespace NS

// This is the implementation of `NS::bar`, which shall be fixed.
void NS::bar(int *p) {
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:20}:"std::span<int> p"
  int tmp;
  tmp = p[5];
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void NS::bar(int *p) {return NS::bar(std::span<int>(p, <# size #>));}\n"

namespace NESTED {
  void alpha(int);
  void beta(int *, int *);

  namespace INNER {
    // `NESTED::INNER::alpha` is distinct from `NESTED::alpha`, so it
    // has no overload and shall be fixed.
    void alpha(int *p) {
      // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:22}:"std::span<int> p"
      int tmp;
      tmp = p[5];
    }
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:6-[[@LINE-1]]:6}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void alpha(int *p) {return alpha(std::span<int>(p, <# size #>));}\n"
  }
}

namespace NESTED {
  // There is an `NESTED::beta(int *, int *)` declared above, so this
  // unsafe function will not be fixed.
  void beta(int *p) {
    // CHECK-NOT: fix-it:"{{.*}}":[[@LINE-1]]:
    int tmp;
    tmp = p[5];
  }

  namespace INNER {
    void delta(int);  // No fix for `NESTED::INNER::delta`
    void delta(int*);
  }
}

// There is an `NESTED::beta(int *)` declared above, so this unsafe
// function will not be fixed.
void NESTED::beta(int *p, int *q) {
  // CHECK-NOT: fix-it:"{{.*}}":[[@LINE-1]]:
  int tmp;
  tmp = p[5];
  tmp = q[5];
}

void NESTED::INNER::delta(int * p) {
  // CHECK-NOT: fix-it:"{{.*}}":[[@LINE-1]]:
  int tmp;
  tmp = p[5];
}


#endif
