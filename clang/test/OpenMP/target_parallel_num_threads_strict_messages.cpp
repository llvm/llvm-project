// RUN: %clang_cc1 -DF1 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -DF1 -DTARGET -verify -fopenmp -fopenmp-version=60 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o /dev/null
// RUN: %clang_cc1 -DF2 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -DF2 -DTARGET -verify -fopenmp -fopenmp-version=60 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o /dev/null
// RUN: %clang_cc1 -DF3 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -DF3 -DTARGET -verify -fopenmp -fopenmp-version=60 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o /dev/null

// RUN: %clang_cc1 -DF1 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-spirv-ppc-host-ppc.bc
// RUN: %clang_cc1 -DF1 -DTARGET -verify -fopenmp -fopenmp-version=60 -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-spirv-ppc-host-ppc.bc -o /dev/null
// RUN: %clang_cc1 -DF2 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-spirv-ppc-host-ppc.bc
// RUN: %clang_cc1 -DF2 -DTARGET -verify -fopenmp -fopenmp-version=60 -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-spirv-ppc-host-ppc.bc -o /dev/null
// RUN: %clang_cc1 -DF3 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-spirv-ppc-host-ppc.bc
// RUN: %clang_cc1 -DF3 -DTARGET -verify -fopenmp -fopenmp-version=60 -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-spirv-ppc-host-ppc.bc -o /dev/null

#ifndef TARGET
// expected-no-diagnostics
#endif

#ifdef F3
template<typename tx>
tx ftemplate(int n) {
  tx a = 0;

#ifdef TARGET
  // expected-warning@+2 {{modifier 'strict' is currently not supported on a GPU for the 'num_threads' clause; modifier ignored}}
#endif
  #pragma omp parallel num_threads(strict: tx(20)) severity(fatal) message("msg")
  {
  }

  short b = 1;
#ifdef TARGET
  // expected-warning@+2 {{modifier 'strict' is currently not supported on a GPU for the 'num_threads' clause; modifier ignored}}
#endif
  #pragma omp parallel num_threads(strict: b) severity(warning) message("msg")
  {
    a += b;
  }

  return a;
}
#endif

#ifdef F2
static
int fstatic(int n) {

#ifdef TARGET
  // expected-warning@+2 {{modifier 'strict' is currently not supported on a GPU for the 'num_threads' clause; modifier ignored}}
#endif
  #pragma omp target parallel num_threads(strict: n) message("msg")
  {
  }

#ifdef TARGET
  // expected-warning@+2 {{modifier 'strict' is currently not supported on a GPU for the 'num_threads' clause; modifier ignored}}
#endif
  #pragma omp target parallel num_threads(strict: 32+n) severity(warning)
  {
  }

  return n+1;
}
#endif

#ifdef F1
struct S1 {
  double a;

  int r1(int n){
    int b = 1;

#ifdef TARGET
    // expected-warning@+2 {{modifier 'strict' is currently not supported on a GPU for the 'num_threads' clause; modifier ignored}}
#endif
    #pragma omp parallel num_threads(strict: n-b) severity(warning) message("msg")
    {
      this->a = (double)b + 1.5;
    }

#ifdef TARGET
    // expected-warning@+2 {{modifier 'strict' is currently not supported on a GPU for the 'num_threads' clause; modifier ignored}}
#endif
    #pragma omp parallel num_threads(strict: 1024) severity(fatal)
    {
      this->a = 2.5;
    }

    return (int)a;
  }
};
#endif

int bar(int n){
  int a = 0;

#ifdef F1
  #pragma omp target
  {
    S1 S;
    a += S.r1(n);
  }
#endif

#ifdef F2
  a += fstatic(n);
#endif

#ifdef F3
  #pragma omp target
  a += ftemplate<int>(n);
#endif

  return a;
}
