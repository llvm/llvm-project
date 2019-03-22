// RUN: %clang_cc1 -fopenmp -x c -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -fsyntax-only -Wuninitialized
// RUN: %clang_cc1 -verify -DDIAGS -DIMMEDIATE -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -fsyntax-only -Wuninitialized
// RUN: %clang_cc1 -verify -DDIAGS -DDELAYED -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -fsyntax-only -Wuninitialized
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

#ifndef DIAGS
// expected-no-diagnostics
#endif // DIAGS

#ifdef IMMEDIATE
#pragma omp declare target
#endif //IMMEDIATE
void t1(int r) {
#ifdef DIAGS
// expected-error@+4 {{invalid input constraint 'mx' in asm}}
#endif // DIAGS
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]"
          : [ r ] "+r"(r)
          : [ lf ] "mx"(0), [ li ] "mr"(0), [ xx ] "x"((double)(0)));
}

unsigned t2(signed char input) {
  unsigned output;
#ifdef DIAGS
// expected-error@+3 {{invalid output constraint '=a' in asm}}
#endif // DIAGS
  __asm__("xyz"
          : "=a"(output)
          : "0"(input));
  return output;
}

double t3(double x) {
  register long double result;
#ifdef DIAGS
// expected-error@+3 {{invalid output constraint '=t' in asm}}
#endif // DIAGS
  __asm __volatile("frndint"
                   : "=t"(result)
                   : "0"(x));
  return result;
}

unsigned char t4(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;
#ifdef DIAGS
// expected-error@+3 {{invalid output constraint '=la' in asm}}
#endif // DIAGS
  __asm__("0:\n1:\n"
          : [ bigres ] "=la"(bigres)
          : [ la ] "0"(la), [ lb ] "c"(lb)
          : "edx", "cc");
  res = bigres;
  return res;
}

void t5(void) {
#ifdef DIAGS
// expected-error@+6 {{unknown register name 'st' in asm}}
#endif // DIAGS
  __asm__ __volatile__(
      "finit"
      :
      :
      : "st", "st(1)", "st(2)", "st(3)",
        "st(4)", "st(5)", "st(6)", "st(7)",
        "fpsr", "fpcr");
}

typedef long long __m256i __attribute__((__vector_size__(32)));
void t6(__m256i *p) {
#ifdef DIAGS
// expected-error@+3 {{unknown register name 'ymm0' in asm}}
#endif // DIAGS
  __asm__ volatile("vmovaps  %0, %%ymm0" ::"m"(*(__m256i *)p)
                   : "ymm0");
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
#ifdef DELAYED
// expected-note@+2 {{called by 'main'}}
#endif // DELAYED
    t2(0);
#ifdef DELAYED
// expected-note@+2 {{called by 'main'}}
#endif // DELAYED
    t3(0);
#ifdef DELAYED
// expected-note@+2 {{called by 'main'}}
#endif // DELAYED
    t4(0, 0);
#ifdef DELAYED
// expected-note@+2 {{called by 'main'}}
#endif // DELAYED
    t5();
#ifdef DELAYED
// expected-note@+2 {{called by 'main'}}
#endif // DELAYED
    t6(0);
  }
  return 0;
}
