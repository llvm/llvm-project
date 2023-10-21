// -fopenmp, -fno-openmp-extensions
// RUN: %clang_cc1 -verify=expected,ge50,lt51,omp,lt51-omp -fopenmp -fno-openmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,omp,lt51-omp -fopenmp -fno-openmp-extensions -fopenmp-version=40 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,omp,lt51-omp -fopenmp -fno-openmp-extensions -fopenmp-version=45 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,lt51,omp,lt51-omp -fopenmp -fno-openmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,ge51,omp,ge51-omp -fopenmp -fno-openmp-extensions -fopenmp-version=51 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,ge51,ge52,omp,ge52-omp -fopenmp -fno-openmp-extensions -fopenmp-version=52 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -DCCODE -verify -fopenmp -fno-openmp-extensions -ferror-limit 300 -x c %s -Wno-openmp -Wuninitialized -Wno-vla

// -fopenmp-simd, -fno-openmp-extensions
// RUN: %clang_cc1 -verify=expected,ge50,lt51,omp,lt51-omp -fopenmp-simd -fno-openmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,omp,lt51-omp -fopenmp-simd -fno-openmp-extensions -fopenmp-version=40 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,omp,lt51-omp -fopenmp-simd -fno-openmp-extensions -fopenmp-version=45 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,lt51,omp,lt51-omp -fopenmp-simd -fno-openmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,ge51,omp,ge51-omp -fopenmp-simd -fno-openmp-extensions -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -DCCODE -verify -fopenmp-simd -fno-openmp-extensions -ferror-limit 300 -x c %s -Wno-openmp-mapping -Wuninitialized -Wno-vla

// -fopenmp -fopenmp-extensions
// RUN: %clang_cc1 -verify=expected,ge50,lt51,ompx,lt51-ompx -fopenmp -fopenmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,ompx,lt51-ompx -fopenmp -fopenmp-extensions -fopenmp-version=40 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,ompx,lt51-ompx -fopenmp -fopenmp-extensions -fopenmp-version=45 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,lt51,ompx,lt51-ompx -fopenmp -fopenmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,ge51,ompx,ge51-ompx -fopenmp -fopenmp-extensions -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -DCCODE -verify -fopenmp -fopenmp-extensions -ferror-limit 300 -x c %s -Wno-openmp -Wuninitialized -Wno-vla

// -fopenmp-simd -fopenmp-extensions
// RUN: %clang_cc1 -verify=expected,ge50,lt51,ompx,lt51-ompx -fopenmp-simd -fopenmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,ompx,lt51-ompx -fopenmp-simd -fopenmp-extensions -fopenmp-version=40 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,lt50,lt51,ompx,lt51-ompx -fopenmp-simd -fopenmp-extensions -fopenmp-version=45 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,lt51,ompx,lt51-ompx -fopenmp-simd -fopenmp-extensions -fopenmp-version=50 -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -verify=expected,ge50,ge51,ompx,ge51-ompx -fopenmp-simd -fopenmp-extensions -ferror-limit 300 %s -Wno-openmp-target -Wuninitialized -Wno-vla
// RUN: %clang_cc1 -DCCODE -verify -fopenmp-simd -fopenmp-extensions -ferror-limit 300 -x c %s -Wno-openmp-mapping -Wuninitialized -Wno-vla

// Check
#ifdef CCODE
void foo(int arg) {
  const int n = 0;

  double marr[10][10][10];

  #pragma omp target map(marr[2][0:2][0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:][0:][:])
  {}
  #pragma omp target map(marr[:][1:][:]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:][n:][:])
  {}
}
#else

void xxx(int argc) {
  int map; // expected-note {{initialize the variable 'map' to silence this warning}}
#pragma omp target map(tofrom: map) // expected-warning {{variable 'map' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}


struct SREF {
  int &a;
  int b;
  SREF(int &a) : a(a) {}
};

template <typename T, int I>
struct SA {
  static int ss;
  #pragma omp threadprivate(ss) // expected-note {{defined as threadprivate or thread local}}
  float a;
  int b[12];
  float *c;
  T d;
  float e[I];
  T *f;
  int bf : 20;
  void func(int arg) {
    SREF sref(arg);
    #pragma omp target
    {
      a = 0.0;
      func(arg);
      bf = 20;
    }
    #pragma omp target map(arg,a,d,sref.b)
    {}
    #pragma omp target map(arg[2:2],a,d) // expected-error {{subscripted value is not an array or pointer}}
    {}
    #pragma omp target map(arg,a*2) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}} ge50-error {{expected addressable lvalue in 'map' clause}}
    {}
    #pragma omp target map(arg,(c+1)[2]) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
    {}
    #pragma omp target map(arg,a[:2],d) // expected-error {{subscripted value is not an array or pointer}}
    {}
    #pragma omp target map(arg,a,d[:2]) // expected-error {{subscripted value is not an array or pointer}}
    {}

    #pragma omp target map(to:ss) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
    {}

    #pragma omp target map(to:b,e)
    {}
    #pragma omp target map(to:b,e) map(to:b) // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
    {}
    #pragma omp target map(to:b[:2],e)
    {}
    #pragma omp target map(to:b,e[:])
    {}
    #pragma omp target map(b[-1:]) // expected-error {{array section must be a subset of the original array}}
    {}
    #pragma omp target map(b[:-1]) // expected-error {{section length is evaluated to a negative value -1}}
    {}
    #pragma omp target map(b[true:true])
    {}

    #pragma omp target map(: c,f) // expected-error {{missing map type}}
    {}
    #pragma omp target map(always, tofrom: c,f)
    {}
    #pragma omp target map(always, tofrom: c[1:2],f)
    {}
    #pragma omp target map(always, tofrom: c,f[1:2])
    {}
    #pragma omp target map(always, tofrom: c[:],f)   // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    {}
    #pragma omp target map(always, tofrom: c,f[:])   // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    {}
    #pragma omp target map(always)   // expected-error {{use of undeclared identifier 'always'}}
    {}
    #pragma omp target map(close, tofrom: c,f)
    {}
    #pragma omp target map(close, tofrom: c[1:2],f)
    {}
    #pragma omp target map(close, tofrom: c,f[1:2])
    {}
    #pragma omp target map(close, tofrom: c[:],f)   // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    {}
    #pragma omp target map(close, tofrom: c,f[:])   // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    {}
    #pragma omp target map(close)   // expected-error {{use of undeclared identifier 'close'}}
    {}
    // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(present, tofrom: c,f)
    {}
    // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(present, tofrom: c[1:2],f)
    {}
    // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(present, tofrom: c,f[1:2])
    {}
    // expected-error@+2 {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(present, tofrom: c[:],f)
    {}
    // expected-error@+2 {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(present, tofrom: c,f[:])
    {}
    // expected-error@+1 {{use of undeclared identifier 'present'}}
    #pragma omp target map(present)
    {}
    // ge52-omp-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(ompx_hold, tofrom: c,f)
    {}
    // ge52-omp-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(ompx_hold, tofrom: c[1:2],f)
    {}
    // ge52-omp-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(ompx_hold, tofrom: c,f[1:2])
    {}
    // ge52-omp-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // expected-error@+3 {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(ompx_hold, tofrom: c[:],f)
    {}
    // ge52-omp-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // expected-error@+3 {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(ompx_hold, tofrom: c,f[:])
    {}
    // expected-error@+1 {{use of undeclared identifier 'ompx_hold'}}
    #pragma omp target map(ompx_hold)
    {}
    #pragma omp target map(close, close, tofrom: a)   // expected-error {{same map type modifier has been specified more than once}}
    {}
    #pragma omp target map(always, close, always, close, tofrom: a)   // expected-error 2 {{same map type modifier has been specified more than once}}
    {}
    // ge51-error@+2 {{same map type modifier has been specified more than once}}
    // lt51-error@+1 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(present, present, tofrom: a)
    {}
    // ge52-omp-error@+5 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // ge52-omp-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // ompx-error@+3 {{same map type modifier has been specified more than once}}
    // ge51-omp-error@+2 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(ompx_hold, ompx_hold, tofrom: a)
    {}
    // ge52-omp-error@+9 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // ge52-omp-error@+8 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'iterator'}}
    // expected-error@+7 2 {{same map type modifier has been specified more than once}}
    // ge51-error@+6 {{same map type modifier has been specified more than once}}
    // lt51-ompx-error@+5 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
    // lt51-omp-error@+4 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    // ompx-error@+3 {{same map type modifier has been specified more than once}}
    // ge51-omp-error@+2 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(always, close, present, ompx_hold, always, close, present, ompx_hold, tofrom: a)
    {}
    #pragma omp target map( , tofrom: a)   // expected-error {{missing map type modifier}}
    {}
    #pragma omp target map( , , tofrom: a)   // expected-error {{missing map type modifier}} expected-error {{missing map type modifier}}
    {}
    #pragma omp target map( , , : a)   // expected-error {{missing map type modifier}} expected-error {{missing map type modifier}} expected-error {{missing map type}}
    {}
    // ge51-error@+3 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-error@+2 2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    // expected-error@+1 {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
    #pragma omp target map( d, f, bf: a)
    {}
    // expected-error@+4 {{missing map type modifier}}
    // ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    // expected-error@+1 {{missing map type}}
    #pragma omp target map( , f, : a)
    {}
    #pragma omp target map(always close: a)   // expected-error {{missing map type}}
    {}
    #pragma omp target map(always close bf: a)   // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
    {}
    // ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    // expected-error@+1 {{missing map type}}
    #pragma omp target map(always tofrom close: a)
    {}
    // ge51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(tofrom from: a)
    {}
    #pragma omp target map(close bf: a)   // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
    {}
    #pragma omp target map(([b[I]][bf])f)  // lt50-error {{expected ',' or ']' in lambda capture list}} lt50-error {{expected ')'}} lt50-note {{to match this '('}}
    {}
    // ge51-ompx-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}}
    // lt51-ompx-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(iterator(it=0:10), tofrom:a)
    {}
    // ompx-error@+8 {{redefinition of 'it'}}
    // ompx-note@+7 {{previous definition is here}}
    // omp-error@+6 {{redefinition of 'it'}}
    // omp-note@+5 {{previous definition is here}}
    // ge51-ompx-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}}
    // lt51-ompx-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(iterator(it=0:10, it=0:20), tofrom:a)
    {}
    // ge51-ompx-error@+6 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}}
    // lt51-ompx-error@+5 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
    // lt51-error@+4 {{expected '(' after 'iterator'}}
    // ge51-error@+3 {{expected '(' after 'iterator'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(iterator, tofrom:a)
    {}
    // ge51-ompx-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}}
    // lt51-ompx-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
    // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
    // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
    #pragma omp target map(iterator(), tofrom:a)
    {}
    return;
  }
};

struct SB {
  unsigned A;
  unsigned B;
  float Arr[100];
  float *Ptr;
  float *foo() {
    return &Arr[0];
  }
};

struct SC {
  unsigned A : 2;
  unsigned B : 3;
  unsigned C;
  unsigned D;
  float Arr[100];
  SB S;
  SB ArrS[100];
  SB *PtrS;
  SB *&RPtrS;
  float *Ptr;

  SC(SB *&_RPtrS) : RPtrS(_RPtrS) {}
};

union SD {
  unsigned A;
  float B;
};

void SAclient(int arg) {
  SA<int,123> s;
  s.func(arg); // expected-note {{in instantiation of member function}}
  double marr[10][10][10];
  double marr2[5][10][1];
  double mvla[5][arg][10];
  double ***mptr;
  const int n = 0;
  const int m = 1;
  double mvla2[5][arg][m+n+10];

  SB *p;

  SD u;
  SC r(p),t(p);
#pragma omp target map(r)
  {}
#pragma omp target map(marr[2] [0:2] [0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[:] [0:2] [0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[2][3] [0:2])
  {}
#pragma omp target map(marr[:][:][:])
  {}
#pragma omp target map(marr[:2][:][:])
  {}
#pragma omp target map(marr [arg:][:][:])
  {}
#pragma omp target map(marr [arg:])
  {}
#pragma omp target map(marr [arg:][:arg][:]) // correct if arg is the size of dimension 2
  {}
#pragma omp target map(marr[:arg][:])
  {}
#pragma omp target map(marr[:arg] [n:])
  {}
#pragma omp target map(marr[:][:arg] [n:]) // correct if arg is the size of  dimension 2
  {}
#pragma omp target map(marr[:][:m] [n:])   // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr [n:m][:arg] [n:])
  {}
#pragma omp target map(marr[:2][:1][:])  // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[:2] [1:][:]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[:2][:][:1])  // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[:2][:] [1:]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[:1][:2][:])
  {}
#pragma omp target map(marr[:1][0][:])
  {}
#pragma omp target map(marr[:arg][:2][:]) // correct if arg is 1
  {}
#pragma omp target map(marr[:1] [3:1][:2])
  {}
#pragma omp target map(marr[:1] [3:arg][:2]) // correct if arg is 1
  {}
#pragma omp target map(marr[:1] [3:2][:2])   // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr[:2][:10][:])
  {}
#pragma omp target map(marr[:2][:][:5 + 5])
  {}
#pragma omp target map(marr[:2] [2 + 2 - 4:] [0:5 + 5])
  {}

#pragma omp target map(marr[:1][:2][0]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(marr2[:1][:2][0])
  {}

#pragma omp target map(mvla[:1][:][0])    // correct if the size of dimension 2 is 1.
  {}
#pragma omp target map(mvla[:2][:arg][:]) // correct if arg is the size of dimension 2.
  {}
#pragma omp target map(mvla[:1][:2][0])   // expected-error {{array section does not specify contiguous storage}}
   {}
#pragma omp target map(mvla[1] [2:arg][:])
  {}
#pragma omp target map(mvla[:1][:][:])
  {}
#pragma omp target map(mvla2[:1][:2][:11])
  {}
#pragma omp target map(mvla2[:1][:2][:10]) // expected-error {{array section does not specify contiguous storage}}
  {}

#pragma omp target map(mptr[:2] [2 + 2 - 4:1] [0:5 + 5]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(mptr[:1][:2 - 1] [2:4 - 3])
  {}
#pragma omp target map(mptr[:1][:arg] [2:4 - 3]) // correct if arg is 1.
  {}
#pragma omp target map(mptr[:1][:2 - 1] [0:2])
  {}
#pragma omp target map(mptr[:1][:2] [0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
#pragma omp target map(mptr[:1][:] [0:2])  // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  {}
#pragma omp target map(mptr[:2][:1] [0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}

#pragma omp target map(r.ArrS[0].B)
  {}
#pragma omp target map(r.ArrS[:1].B)   // expected-error {{OpenMP array section is not allowed here}}
  {}
#pragma omp target map(r.ArrS[:arg].B) // expected-error {{OpenMP array section is not allowed here}}
  {}
#pragma omp target map(r.ArrS[0].Arr [1:23])
  {}
#pragma omp target map(r.ArrS[0].Arr [1:arg])
  {}
#pragma omp target map(r.ArrS[0].Arr [arg:23])
  {}
#pragma omp target map(r.ArrS[0].Error)          // expected-error {{no member named 'Error' in 'SB'}}
  {}
#pragma omp target map(r.ArrS[0].A, r.ArrS[1].A) // lt50-error {{multiple array elements associated with the same variable are not allowed in map clauses of the same construct}} lt50-note {{used here}}
  {}
#pragma omp target map(r.ArrS[0].A, t.ArrS[1].A)
  {}
#pragma omp target map(r.PtrS[0], r.PtrS->B) // lt50-error {{same pointer dereferenced in multiple different ways in map clause expressions}} lt50-note {{used here}}
  {}
#pragma omp target map(r.PtrS, r.PtrS->B)    // lt50-error {{pointer cannot be mapped along with a section derived from itself}} lt50-note {{used here}}
  {}
#pragma omp target map(r.PtrS->A, r.PtrS->B)
  {}
#pragma omp target map(r.RPtrS[0], r.RPtrS->B) // lt50-error {{same pointer dereferenced in multiple different ways in map clause expressions}} lt50-note {{used here}}
  {}
#pragma omp target map(r.RPtrS, r.RPtrS->B)    // lt50-error {{pointer cannot be mapped along with a section derived from itself}} lt50-note {{used here}}
  {}
#pragma omp target map(r.RPtrS->A, r.RPtrS->B)
  {}
#pragma omp target map(r.S.Arr[:12])
  {}
#pragma omp target map(r.S.foo() [:12]) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}} ge50-error {{expected addressable lvalue in 'map' clause}}
  {}
#pragma omp target map(r.C, r.D)
  {}
#pragma omp target map(r.C, r.C)     // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
  {}
#pragma omp target map(r.C) map(r.C) // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
  {}
#pragma omp target map(r.C, r.S)     // this would be an error only caught at runtime - Sema would have to make sure there is not way for the missing data between fields to be mapped somewhere else.
  {}
#pragma omp target map(r, r.S)       // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
  {}
#pragma omp target map(r.C, t.C)
  {}
#pragma omp target map(r.A) // expected-error {{bit fields cannot be used to specify storage in a 'map' clause}}
  {}
#pragma omp target map(r.Arr)
  {}
#pragma omp target map(r.Arr [3:5])
  {}
#pragma omp target map(r.Ptr [3:5])
  {}
#pragma omp target map(r.ArrS [3:5].A)         // expected-error {{OpenMP array section is not allowed here}}
  {}
#pragma omp target map(r.ArrS [3:5].Arr [6:7]) // expected-error {{OpenMP array section is not allowed here}}
  {}
#pragma omp target map(r.ArrS[3].Arr [6:7])
  {}
#pragma omp target map(r.S.Arr [4:5])
  {}
#pragma omp target map(r.S.Ptr [4:5])
  {}
#pragma omp target map(r.S.Ptr[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  {}
#pragma omp target map((p + 1)->A) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(u.B)        // expected-error {{mapping of union members is not allowed}}
  {}
#pragma omp target
  {
    u.B = 0;
    r.S.foo();
  }

#pragma omp target data map(to \
                            : r.C) // lt50-note {{used here}}
  {
#pragma omp target map(r.D)        // lt50-error {{original storage of expression in data environment is shared but data environment do not fully contain mapped expression storage}}
    {}
  }

#pragma omp target data map(to \
                            : t.Ptr) // lt50-note {{used here}}
  {
#pragma omp target map(t.Ptr[:23])   // lt50-error {{pointer cannot be mapped along with a section derived from itself}}
    {}
  }

#pragma omp target data map(to \
                            : t.C, t.D)
  {
#pragma omp target data map(to \
                            : t.C)
  {
#pragma omp target map(t.D)
    {}
  }
  }
#pragma omp target data map(marr[:][:][:])
  {
#pragma omp target data map(marr)
    {}
  }

#pragma omp target data map(to \
                            : t)
  {
#pragma omp target data map(to \
                            : t.C)
  {
#pragma omp target map(t.D)
    {}
  }
  }
}
void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}} // expected-note 3 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4);
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { }
public:
  S5(int v):a(v) { }
};

template <class T>
struct S6;

template<>
struct S6<int>
{
   virtual void foo();
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

typedef int from;

struct dim {
  double x, y;
};

template<typename T>
class Array1D
{
  public:
    unsigned n1;
    unsigned size;
    T * dptr;

    inline T& operator() (unsigned i1) { return dptr[i1]; }

    Array1D() {n1=0;size=0;dptr=nullptr;}
};

template <typename T, int I> // expected-note {{declared here}}
T tmain(T argc) {
  const T d = 5;
  const T da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  T i, t[20];
  T &j = i;
  T *k = &j;
  T x;
  T y;
  T to, tofrom, always, close, present;
  const T (&l)[5] = da;
#pragma omp target map // expected-error {{expected '(' after 'map'}}
  {}
#pragma omp target map( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  {}
#pragma omp target map() // expected-error {{expected expression}}
  {}
#pragma omp target map(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  {}
#pragma omp target map(to argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ',' or ')' in 'map' clause}}
  {}
#pragma omp target map(to:) // expected-error {{expected expression}}
  {}
#pragma omp target map(from: argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
#pragma omp target map(x: y) // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
  {}
#pragma omp target map(x)
  foo();
#pragma omp target map(tofrom: t[:I])
  foo();
#pragma omp target map(T: a) // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}} expected-error {{incomplete type 'S1' where a complete type is required}}
  foo();
#pragma omp target map(T) // expected-error {{'T' does not refer to a value}}
  foo();
#pragma omp target map(I) // lt50-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}} ge50-error 2 {{expected addressable lvalue in 'map' clause}}
  foo();
#pragma omp target map(S2::S2s)
  foo();
#pragma omp target map(S2::S2sc)
  foo();
#pragma omp target map(x)
  foo();
#pragma omp target map(to: x)
  foo();
#pragma omp target map(to: to)
  foo();
#pragma omp target map(to)
  foo();
#pragma omp target map(to, x)
  foo();
#pragma omp target data map(to x) // expected-error {{expected ',' or ')' in 'map' clause}}
#pragma omp target data map(tofrom: argc > 0 ? x : y) // lt50-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}} ge50-error 2 {{expected addressable lvalue in 'map' clause}}
#pragma omp target data map(argc)
#pragma omp target data map(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target data map(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}} warn-warning 2 {{Type 'const S2' is not trivially copyable and not guaranteed to be mapped correctly}} warn-warning 2 {{Type 'const S3' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(ba) // warn-warning 2 {{Type 'const S2 [5]' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(ca) // warn-warning 2 {{Type 'const S3 [5]' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(da)
#pragma omp target data map(S2::S2s)
#pragma omp target data map(S2::S2sc)
#pragma omp target data map(e, g) // warn-warning 2 {{Type 'S4' is not trivially copyable and not guaranteed to be mapped correctly}} warn-warning 2 {{Type 'S5' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(h) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
#pragma omp target data map(k) map(k) // lt50-error 2 {{variable already marked as mapped in current construct}} lt50-note 2 {{used here}}
#pragma omp target map(k), map(k[:5]) // lt50-error 2 {{pointer cannot be mapped along with a section derived from itself}} lt50-note 2 {{used here}}
  foo();
#pragma omp target data map(da)
#pragma omp target map(da[:4])
  foo();
#pragma omp target data map(k, j, l) // lt50-note 2 {{used here}}
#pragma omp target data map(k[:4]) // lt50-error 2 {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l) map(l[:5]) // lt50-error 2 {{variable already marked as mapped in current construct}} lt50-note 2 {{used here}}
  foo();
#pragma omp target data map(k[:4], j, l[:5]) // lt50-note 2 {{used here}}
#pragma omp target data map(k) // lt50-error 2 {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l)
  foo();

#pragma omp target data map(always, tofrom: x)
#pragma omp target data map(always: x) // expected-error {{missing map type}}
// ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// expected-error@+1 {{missing map type}}
#pragma omp target data map(tofrom, always: x)
#pragma omp target data map(always, tofrom: always, tofrom, x)
#pragma omp target map(tofrom j) // expected-error {{expected ',' or ')' in 'map' clause}}
  foo();

#pragma omp target data map(close, tofrom: x)
#pragma omp target data map(close: x) // expected-error {{missing map type}}
// ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// expected-error@+1 {{missing map type}}
#pragma omp target data map(tofrom, close: x)
#pragma omp target data map(close, tofrom: close, tofrom, x)
  foo();

// lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target data map(present, tofrom: x)
// ge51-error@+2 {{missing map type}}
// lt51-error@+1 {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target data map(present: x)
// ge51-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// ge51-error@+2 {{missing map type}}
// lt51-error@+1 {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target data map(tofrom, present: x)
// lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target data map(present, tofrom: present, tofrom, x)
  foo();

  T marr[10][10], iarr[5];
#pragma omp target data map(marr[10][0:2:2]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  {}
#pragma omp target data map(iarr[:2:d]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  {}
  return 0;
}

struct SA1{
  int a;
  struct SA1 *p;
  int b[10];
};
struct SB1{
  int a;
  struct SA1 s;
  struct SA1 sa[10];
  struct SA1 *sp[10];
  struct SA1 *p;
};
struct SC1{
  int a;
  struct SB1 s;
  struct SB1 *p;
  int b[10];
};

class S8 {
public:
  virtual void foo() = 0;
} *s8;

class S9 {
public:
  virtual void foo() {}
} s9;


int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;
  int *k = &j;
  S6<int> m;
  int x;
  int y;
  int to, tofrom, always, close, present;
  const int (&l)[5] = da;
  SC1 s;
  SC1 *p;
  int Arr[10];
#pragma omp target data map // expected-error {{expected '(' after 'map'}} lt50-error {{expected at least one 'map' or 'use_device_ptr' clause for '#pragma omp target data'}} ge50-error {{expected at least one 'map', 'use_device_ptr', or 'use_device_addr' clause for '#pragma omp target data'}}
#pragma omp target data map( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
#pragma omp target data map() // expected-error {{expected expression}}
#pragma omp target data map(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
#pragma omp target data map(to argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ',' or ')' in 'map' clause}}
#pragma omp target data map(to:) // expected-error {{expected expression}}
#pragma omp target data map(from: argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target data map(x: y) // expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target map(x)
  foo();
#pragma omp target map(to: x)
  foo();
#pragma omp target map(to: to)
  foo();
#pragma omp target map(to)
  foo();
#pragma omp target map(to, x)
  foo();
#pragma omp target data map(to x) // expected-error {{expected ',' or ')' in 'map' clause}}
#pragma omp target data map(tofrom: argc > 0 ? argv[1] : argv[2]) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}} ge50-error {{expected addressable lvalue in 'map' clause}}
#pragma omp target data map(argc)
#pragma omp target data map(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target data map(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}} warn-warning {{Type 'const S2' is not trivially copyable and not guaranteed to be mapped correctly}} warn-warning {{Type 'const S3' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(argv[1])
#pragma omp target data map(ba) // warn-warning {{Type 'const S2 [5]' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(ca) // warn-warning {{Type 'const S3 [5]' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(da)
#pragma omp target data map(S2::S2s)
#pragma omp target data map(S2::S2sc)
#pragma omp target data map(e, g) // warn-warning {{Type 'S4' is not trivially copyable and not guaranteed to be mapped correctly}} warn-warning {{Type 'S5' is not trivially copyable and not guaranteed to be mapped correctly}}
#pragma omp target data map(h) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
#pragma omp target data map(k), map(k) // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
#pragma omp target map(k), map(k[:5]) // lt50-error {{pointer cannot be mapped along with a section derived from itself}} lt50-note {{used here}}
  foo();
#pragma omp target data map(da)
#pragma omp target map(da[:4])
  foo();
#pragma omp target data map(k, j, l) // lt50-note {{used here}}
#pragma omp target data map(k[:4]) // lt50-error {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l) map(l[:5]) // lt50-error {{variable already marked as mapped in current construct}} lt50-note {{used here}}
  foo();
#pragma omp target data map(k[:4], j, l[:5]) // lt50-note {{used here}}
#pragma omp target data map(k) // lt50-error {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l)
  foo();

#pragma omp target data map(always, tofrom: x)
#pragma omp target data map(always: x) // expected-error {{missing map type}}
// ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// expected-error@+1 {{missing map type}}
#pragma omp target data map(tofrom, always: x)
#pragma omp target data map(always, tofrom: always, tofrom, x)
#pragma omp target map(tofrom j) // expected-error {{expected ',' or ')' in 'map' clause}}
  foo();
#pragma omp target data map(close, tofrom: x)
#pragma omp target data map(close: x) // expected-error {{missing map type}}
// ge51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// expected-error@+1 {{missing map type}}
#pragma omp target data map(tofrom, close: x)
  foo();
// lt51-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
#pragma omp target data map(present, tofrom: x)
// ge51-error@+2 {{missing map type}}
// lt51-error@+1 {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target data map(present: x)
// ge51-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
// lt51-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
// ge51-error@+2 {{missing map type}}
// lt51-error@+1 {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target data map(tofrom, present: x)
  foo();
#pragma omp target private(j) map(j) // expected-error {{private variable cannot be in a map clause in '#pragma omp target' directive}}  expected-note {{defined as private}}
  {}
#pragma omp target firstprivate(j) map(j)  // expected-error {{firstprivate variable cannot be in a map clause in '#pragma omp target' directive}} expected-note {{defined as firstprivate}}
  {}
#pragma omp target map(m) // warn-warning {{Type 'S6<int>' is not trivially copyable and not guaranteed to be mapped correctly}}
  {}
#pragma omp target
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.s)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.s.a)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.b[:5])
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
#pragma omp target map(s.p[:5])
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.sa[3].a)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.sp[3]->a)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.p->a)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.p->a)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.s.b[:2])
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.s.p->b[:2])
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
// lt50-note@+1 {{used here}}
#pragma omp target map(s.p->p->p->a)
// lt50-error@+1 {{variable already marked as mapped in current construct}}
  { s.a++; }
#pragma omp target map(s.s.s.b[:2])
  { s.s.s.b[0]++; }
#pragma omp target map(s8[0:1], s9) // warn-warning {{Type 'class S8' is not trivially copyable and not guaranteed to be mapped correctly}} warn-warning {{Type 'class S9' is not trivially copyable and not guaranteed to be mapped correctly}}
  {}

  int **BB, *offset, *a;

#pragma omp target map(**(BB+*offset)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(**(BB+y)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(*(a+*offset)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(**(*offset+BB)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(**(y+BB)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(*(*offset+a)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(**(*offset+BB+*a)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(**(*(*(&offset))+BB+*a)) // lt50-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
#pragma omp target map(*(a+(a))) // expected-error {{invalid operands to binary expression ('int *' and 'int *')}}
  {}
#pragma omp target map(*(1+*a+*a)) // expected-error {{indirection requires pointer operand ('int' invalid)}}
  {}

#pragma omp target map(delete: a) // expected-error {{map type 'delete' is not allowed for '#pragma omp target'}}
  {}
#pragma omp target map(release: a) // expected-error {{map type 'release' is not allowed for '#pragma omp target'}}
  {}

  int marr[10][10], iarr[5];

#pragma omp target map(marr[10][0:2:2]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  {}
#pragma omp target map(iarr[:2:d]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  {}

#pragma omp target data map(Arr[0:4]) // lt50-note {{used here}}
  {
#pragma omp target
    Arr[0] = 2; // lt50-error {{original storage of expression in data environment is shared but data environment do not fully contain mapped expression storage}}
  }

  Array1D<dim> pos;

#pragma omp target enter data map(to:pos)
#pragma omp target enter data map(to:pos.dptr[0:pos.size])
#pragma omp target teams distribute parallel for
  for(int i=0; i<100; i++) {
    pos(i).x = i;
    pos(i).y = i+1;
  }

  // ge51-ompx-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}}
  // lt51-ompx-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
  // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
  // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  #pragma omp target map(iterator(it=0:10), tofrom:a[it])
  {}

  // ompx-error@+8 {{use of undeclared identifier 'itt'; did you mean 'it'?}}
  // ompx-note@+7 {{'it' declared here}}
  // omp-error@+6 {{use of undeclared identifier 'itt'; did you mean 'it'?}}
  // omp-note@+5 {{'it' declared here}}
  // ge51-ompx-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}}
  // lt51-ompx-error@+3 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}}
  // ge51-omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present'}}
  // lt51-omp-error@+1 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  #pragma omp target map(iterator(it=0:10), tofrom:a[itt])
  {}

  return tmain<int, 3>(argc)+tmain<from, 4>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 3>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<int, 4>' requested here}}
}
#endif
