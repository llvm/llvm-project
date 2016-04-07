// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c++ -std=c++11 -fms-extensions %s

// expected-error@+1 {{expected an OpenMP directive}}
#pragma omp declare

// expected-error@+2 {{'#pragma omp declare simd' can only be applied to functions}}
#pragma omp declare simd
int a;
// expected-error@+2 {{'#pragma omp declare simd' can only be applied to functions}}
#pragma omp declare simd
#pragma omp threadprivate(a)
int var;
#pragma omp threadprivate(var)

// expected-error@+2 {{expected an OpenMP directive}} expected-error@+1 {{function declaration is expected after 'declare simd' directive}}
#pragma omp declare simd
#pragma omp declare

// expected-error@+3 {{function declaration is expected after 'declare simd' directive}}
// expected-error@+1 {{function declaration is expected after 'declare simd' directive}}
#pragma omp declare simd
#pragma omp declare simd
#pragma options align=packed
int main();

// expected-error@+3 {{function declaration is expected after 'declare simd' directive}}
// expected-error@+1 {{function declaration is expected after 'declare simd' directive}}
#pragma omp declare simd
#pragma omp declare simd
#pragma init_seg(compiler)
int main();

// expected-error@+1 {{single declaration is expected after 'declare simd' directive}}
#pragma omp declare simd
// expected-note@+1 {{declared here}}
int b, c;

// expected-error@+1 {{'C' does not refer to a value}}
#pragma omp declare simd simdlen(C)
// expected-note@+1 {{declared here}}
template <class C>
void h(C *hp, C *hp2, C *hq, C *lin) {
  b = 0;
}

#pragma omp declare simd
template <>
void h(int *hp, int *hp2, int *hq, int *lin) {
  h((float *)hp, (float *)hp2, (float *)hq, (float *)lin);
}

#pragma omp declare simd inbranch inbranch
#pragma omp declare simd notinbranch notinbranch
#pragma omp declare simd inbranch inbranch notinbranch // expected-error {{unexpected 'notinbranch' clause, 'inbranch' is specified already}}
#pragma omp declare simd notinbranch notinbranch inbranch // expected-error {{unexpected 'inbranch' clause, 'notinbranch' is specified already}}
// expected-note@+2 {{read of non-const variable 'b' is not allowed in a constant expression}}
// expected-error@+1 {{expression is not an integral constant expression}}
#pragma omp declare simd simdlen(b)
// expected-error@+1 {{directive '#pragma omp declare simd' cannot contain more than one 'simdlen' clause}}
#pragma omp declare simd simdlen(32) simdlen(c)
// expected-error@+1 {{expected '(' after 'simdlen'}}
#pragma omp declare simd simdlen
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected ')'}}
// expected-error@+1 {{expected expression}}
#pragma omp declare simd simdlen(
// expected-error@+2 {{expected '(' after 'simdlen'}}
// expected-error@+1 {{expected expression}}
#pragma omp declare simd simdlen(), simdlen
// expected-error@+1 2 {{expected expression}}
#pragma omp declare simd simdlen(), simdlen()
// expected-warning@+3 {{extra tokens at the end of '#pragma omp declare simd' are ignored}}
// expected-error@+2 {{expected '(' after 'simdlen'}}
// expected-error@+1 {{expected expression}}
#pragma omp declare simd simdlen() simdlen)
void foo();

// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp declare simd simdlen(N)
template<int N>
void foo() {}

void test() {
  // expected-note@+1 {{in instantiation of function template specialization 'foo<-3>' requested here}}
  foo<-3>();
}

template <class T>
struct St {
// expected-error@+2 {{function declaration is expected after 'declare simd' directive}}
#pragma init_seg(compiler)
#pragma omp declare simd
#pragma init_seg(compiler)
  void h(T *hp) {
// expected-error@+1 {{unexpected OpenMP directive '#pragma omp declare simd'}}
#pragma omp declare simd
    *hp = *t;
  }

private:
  T t;
};

namespace N {
  // expected-error@+1 {{function declaration is expected after 'declare simd' directive}}
  #pragma omp declare simd
}
// expected-error@+1 {{function declaration is expected after 'declare simd' directive}}
#pragma omp declare simd
// expected-error@+1 {{function declaration is expected after 'declare simd' directive}}
#pragma omp declare simd
