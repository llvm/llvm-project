// RUN: %clangxx_nsan -O2 -g %s -o %t
// RUN: env NSAN_OPTIONS=check_cmp=true,halt_on_error=0 %run %t 2>&1 | FileCheck %s -check-prefix=CMP
// RUN: env NSAN_OPTIONS=check_cmp=false,halt_on_error=0 %run %t 2>&1 | FileCheck %s --allow-empty

#include <cmath>
#include <cstdio>

// 0.6/0.2 is slightly below 3, so the comparison will fail after a certain
// threshold that depends on the precision of the computation.
__attribute__((noinline))  // To check call stack reporting.
bool DoCmp(double a, double b, double c, double threshold) {
  return c - a / b < threshold;
  // CMP: WARNING: NumericalStabilitySanitizer: floating-point comparison results depend on precision
  // CMP: double    {{ *}}precision dec (native): {{.*}}<{{.*}}
  // CMP: __float128{{ *}}precision dec (shadow): {{.*}}<{{.*}}
  // CMP: {{#0 .*in DoCmp}}
}

int main() {
  double threshold = 1.0;
  for (int i = 0; i < 60; ++i) {
    threshold /= 2;
    printf("value at threshold %.20f: %i\n", threshold,
           DoCmp(0.6, 0.2, 3.0, threshold));
  }
  return 0;
}
