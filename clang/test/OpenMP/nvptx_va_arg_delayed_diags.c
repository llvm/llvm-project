// RUN: %clang_cc1 -fopenmp -x c -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -fsyntax-only
// RUN: %clang_cc1 -verify -DDIAGS -DIMMEDIATE -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -fsyntax-only
// RUN: %clang_cc1 -verify -DDIAGS -DDELAYED -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -fsyntax-only
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

#ifndef DIAGS
// expected-no-diagnostics
#endif // DIAGS

#ifdef IMMEDIATE
#pragma omp declare target
#endif //IMMEDIATE
void t1(int r, ...) {
#ifdef DIAGS
// expected-error@+4 {{CUDA device code does not support va_arg}}
#endif // DIAGS
  __builtin_va_list list;
  __builtin_va_start(list, r);
  (void)__builtin_va_arg(list, int);
  __builtin_va_end(list);
}

#ifdef IMMEDIATE
#pragma omp end declare target
#endif //IMMEDIATE

int main() {
#ifdef DELAYED
#pragma omp target
#endif // DELAYED
  {
#ifdef DELAYED
// expected-note@+2 {{called by 'main'}}
#endif // DELAYED
    t1(0);
  }
  return 0;
}
