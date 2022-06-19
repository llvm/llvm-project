// RUN: %clang_cc1 -fapprox-func  %s -emit-llvm -o - | FileCheck --check-prefix=CHECK-AFN %s
// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck --check-prefix=CHECK-NO-AFN %s

extern double exp(double);
double afn_option_test(double x) {
  return exp(x);
  // CHECK-LABEL:  define{{.*}} double @afn_option_test(double %x) #0 {

  // CHECK-AFN:      %{{.*}} = call afn double @{{.*}}exp{{.*}}(double %{{.*}})
  // CHECK-AFN:      attributes #0 ={{.*}} "approx-func-fp-math"="true" {{.*}}

  // CHECK-NO-AFN:   %{{.*}} = call double @{{.*}}exp{{.*}}(double %{{.*}})
  // CHECK-NO-AFN-NOT:  attributes #0 ={{.*}} "approx-func-fp-math"="true" {{.*}}
}
