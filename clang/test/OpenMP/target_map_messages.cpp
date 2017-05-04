// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 200 %s
// RUN: %clang_cc1 -DCCODE -verify -fopenmp -ferror-limit 200 -x c %s
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
  void func(int arg) {
    #pragma omp target map(arg,a,d)
    {}
    #pragma omp target map(arg[2:2],a,d) // expected-error {{subscripted value is not an array or pointer}}
    {}
    #pragma omp target map(arg,a*2) // expected-error {{expected expression containing only member accesses and/or array sections based on named variables}}
    {}
    #pragma omp target map(arg,(c+1)[2]) // expected-error {{expected expression containing only member accesses and/or array sections based on named variables}}
    {}
    #pragma omp target map(arg,a[:2],d) // expected-error {{subscripted value is not an array or pointer}}
    {}
    #pragma omp target map(arg,a,d[:2]) // expected-error {{subscripted value is not an array or pointer}}
    {}

    #pragma omp target map(to:ss) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
    {}

    #pragma omp target map(to:b,e)
    {}
    #pragma omp target map(to:b,e) map(to:b) // expected-error {{variable already marked as mapped in current construct}} expected-note {{used here}}
    {}
    #pragma omp target map(to:b[:2],e)
    {}
    #pragma omp target map(to:b,e[:])
    {}
    #pragma omp target map(b[-1:]) // expected-error {{array section must be a subset of the original array}}
    {}
    #pragma omp target map(b[:-1]) // expected-error {{section length is evaluated to a negative value -1}}
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
  #pragma omp target map(marr[2][0:2][0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:][0:2][0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[2][3][0:2])
  {}
  #pragma omp target map(marr[:][:][:])
  {}
  #pragma omp target map(marr[:2][:][:])
  {}
  #pragma omp target map(marr[arg:][:][:])
  {}
  #pragma omp target map(marr[arg:])
  {}
  #pragma omp target map(marr[arg:][:arg][:]) // correct if arg is the size of dimension 2
  {}
  #pragma omp target map(marr[:arg][:])
  {}
  #pragma omp target map(marr[:arg][n:])
  {}
  #pragma omp target map(marr[:][:arg][n:]) // correct if arg is the size of  dimension 2
  {}
  #pragma omp target map(marr[:][:m][n:]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[n:m][:arg][n:])
  {}
  #pragma omp target map(marr[:2][:1][:]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:2][1:][:]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:2][:][:1]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:2][:][1:]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:1][:2][:])
  {}
  #pragma omp target map(marr[:1][0][:])
  {}
  #pragma omp target map(marr[:arg][:2][:]) // correct if arg is 1
  {}
  #pragma omp target map(marr[:1][3:1][:2])
  {}
  #pragma omp target map(marr[:1][3:arg][:2]) // correct if arg is 1
  {}
  #pragma omp target map(marr[:1][3:2][:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr[:2][:10][:])
  {}
  #pragma omp target map(marr[:2][:][:5+5])
  {}
  #pragma omp target map(marr[:2][2+2-4:][0:5+5])
  {}

  #pragma omp target map(marr[:1][:2][0]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(marr2[:1][:2][0])
  {}

  #pragma omp target map(mvla[:1][:][0]) // correct if the size of dimension 2 is 1.
  {}
  #pragma omp target map(mvla[:2][:arg][:]) // correct if arg is the size of dimension 2.
  {}
  #pragma omp target map(mvla[:1][:2][0]) // expected-error {{array section does not specify contiguous storage}}
   {}
  #pragma omp target map(mvla[1][2:arg][:])
  {}
  #pragma omp target map(mvla[:1][:][:])
  {}
  #pragma omp target map(mvla2[:1][:2][:11])
  {}
  #pragma omp target map(mvla2[:1][:2][:10]) // expected-error {{array section does not specify contiguous storage}}
  {}

  #pragma omp target map(mptr[:2][2+2-4:1][0:5+5]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(mptr[:1][:2-1][2:4-3])
  {}
  #pragma omp target map(mptr[:1][:arg][2:4-3]) // correct if arg is 1.
  {}
  #pragma omp target map(mptr[:1][:2-1][0:2])
  {}
  #pragma omp target map(mptr[:1][:2][0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}
  #pragma omp target map(mptr[:1][:][0:2]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  {}
  #pragma omp target map(mptr[:2][:1][0:2]) // expected-error {{array section does not specify contiguous storage}}
  {}

  #pragma omp target map(r.ArrS[0].B)
  {}
  #pragma omp target map(r.ArrS[:1].B) // expected-error {{OpenMP array section is not allowed here}}
  {}
  #pragma omp target map(r.ArrS[:arg].B) // expected-error {{OpenMP array section is not allowed here}}
  {}
  #pragma omp target map(r.ArrS[0].Arr[1:23])
  {}
  #pragma omp target map(r.ArrS[0].Arr[1:arg])
  {}
  #pragma omp target map(r.ArrS[0].Arr[arg:23])
  {}
  #pragma omp target map(r.ArrS[0].Error) // expected-error {{no member named 'Error' in 'SB'}}
  {}
  #pragma omp target map(r.ArrS[0].A, r.ArrS[1].A) // expected-error {{multiple array elements associated with the same variable are not allowed in map clauses of the same construct}} expected-note {{used here}}
  {}
  #pragma omp target map(r.ArrS[0].A, t.ArrS[1].A)
  {}
  #pragma omp target map(r.PtrS[0], r.PtrS->B) // expected-error {{same pointer derreferenced in multiple different ways in map clause expressions}} expected-note {{used here}}
  {}
  #pragma omp target map(r.RPtrS[0], r.RPtrS->B) // expected-error {{same pointer derreferenced in multiple different ways in map clause expressions}} expected-note {{used here}}
  {}
  #pragma omp target map(r.S.Arr[:12])
  {}
  #pragma omp target map(r.S.foo()[:12]) // expected-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
  #pragma omp target map(r.C, r.D)
  {}
  #pragma omp target map(r.C, r.C) // expected-error {{variable already marked as mapped in current construct}} expected-note {{used here}}
  {}
  #pragma omp target map(r.C) map(r.C) // expected-error {{variable already marked as mapped in current construct}} expected-note {{used here}}
  {}
  #pragma omp target map(r.C, r.S)  // this would be an error only caught at runtime - Sema would have to make sure there is not way for the missing data between fields to be mapped somewhere else.
  {}
  #pragma omp target map(r, r.S)  // expected-error {{variable already marked as mapped in current construct}} expected-note {{used here}}
  {}
  #pragma omp target map(r.C, t.C)
  {}
  #pragma omp target map(r.A)   // expected-error {{bit fields cannot be used to specify storage in a 'map' clause}}
  {}
  #pragma omp target map(r.Arr)
  {}
  #pragma omp target map(r.Arr[3:5])
  {}
  #pragma omp target map(r.Ptr[3:5])
  {}
  #pragma omp target map(r.ArrS[3:5].A)   // expected-error {{OpenMP array section is not allowed here}}
  {}
  #pragma omp target map(r.ArrS[3:5].Arr[6:7])   // expected-error {{OpenMP array section is not allowed here}}
  {}
  #pragma omp target map(r.ArrS[3].Arr[6:7])
  {}
  #pragma omp target map(r.S.Arr[4:5])
  {}
  #pragma omp target map(r.S.Ptr[4:5])
  {}
  #pragma omp target map(r.S.Ptr[:])  // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  {}
  #pragma omp target map((p+1)->A)  // expected-error {{expected expression containing only member accesses and/or array sections based on named variables}}
  {}
  #pragma omp target map(u.B)  // expected-error {{mapped storage cannot be derived from a union}}
  {}

  #pragma omp target data map(to: r.C) //expected-note {{used here}}
  {
    #pragma omp target map(r.D)  // expected-error {{original storage of expression in data environment is shared but data environment do not fully contain mapped expression storage}}
    {}
  }

  #pragma omp target data map(to: t.Ptr) //expected-note {{used here}}
  {
    #pragma omp target map(t.Ptr[:23])  // expected-error {{pointer cannot be mapped along with a section derived from itself}}
    {}
  }

  #pragma omp target data map(to: t.C, t.D)
  {
  #pragma omp target data map(to: t.C)
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

  #pragma omp target data map(to: t)
  {
  #pragma omp target data map(to: t.C)
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

struct S1; // expected-note 2 {{declared here}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
  static float S2s; // expected-note 4 {{mappable type cannot contain static members}}
  static const float S2sc; // expected-note 4 {{mappable type cannot contain static members}}
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
struct S6<int>  // expected-note {{mappable type cannot be polymorphic}}
{
   virtual void foo();
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

typedef int from;

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
  T to, tofrom, always;
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
#pragma omp target map(I) // expected-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}}
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
#pragma omp target data map(tofrom: argc > 0 ? x : y) // expected-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target data map(argc)
#pragma omp target data map(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target data map(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}} expected-error 2 {{type 'S2' is not mappable to target}}
#pragma omp target data map(ba) // expected-error 2 {{type 'S2' is not mappable to target}}
#pragma omp target data map(ca)
#pragma omp target data map(da)
#pragma omp target data map(S2::S2s)
#pragma omp target data map(S2::S2sc)
#pragma omp target data map(e, g)
#pragma omp target data map(h) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
#pragma omp target data map(k) map(k) // expected-error 2 {{variable already marked as mapped in current construct}} expected-note 2 {{used here}}
#pragma omp target map(k), map(k[:5]) // expected-error 2 {{pointer cannot be mapped along with a section derived from itself}} expected-note 2 {{used here}}
  foo();
#pragma omp target data map(da)
#pragma omp target map(da[:4])
  foo();
#pragma omp target data map(k, j, l) // expected-note 2 {{used here}}
#pragma omp target data map(k[:4]) // expected-error 2 {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l) map(l[:5]) // expected-error 2 {{variable already marked as mapped in current construct}} expected-note 2 {{used here}}
  foo();
#pragma omp target data map(k[:4], j, l[:5]) // expected-note 2 {{used here}}
#pragma omp target data map(k) // expected-error 2 {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l)
  foo();

#pragma omp target data map(always, tofrom: x)
#pragma omp target data map(always: x) // expected-error {{missing map type}}
#pragma omp target data map(tofrom, always: x) // expected-error {{incorrect map type modifier, expected 'always'}} expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target data map(always, tofrom: always, tofrom, x)
#pragma omp target map(tofrom j) // expected-error {{expected ',' or ')' in 'map' clause}}
  foo();
  return 0;
}

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
  int to, tofrom, always;
  const int (&l)[5] = da;
#pragma omp target data map // expected-error {{expected '(' after 'map'}} expected-error {{expected at least one map clause for '#pragma omp target data'}}
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
#pragma omp target data map(tofrom: argc > 0 ? argv[1] : argv[2]) // expected-error {{xpected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target data map(argc)
#pragma omp target data map(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target data map(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}} expected-error 2 {{type 'S2' is not mappable to target}}
#pragma omp target data map(argv[1])
#pragma omp target data map(ba) // expected-error 2 {{type 'S2' is not mappable to target}}
#pragma omp target data map(ca)
#pragma omp target data map(da)
#pragma omp target data map(S2::S2s)
#pragma omp target data map(S2::S2sc)
#pragma omp target data map(e, g)
#pragma omp target data map(h) // expected-error {{threadprivate variables are not allowed in 'map' clause}}
#pragma omp target data map(k), map(k) // expected-error {{variable already marked as mapped in current construct}} expected-note {{used here}}
#pragma omp target map(k), map(k[:5]) // expected-error {{pointer cannot be mapped along with a section derived from itself}} expected-note {{used here}}
  foo();
#pragma omp target data map(da)
#pragma omp target map(da[:4])
  foo();
#pragma omp target data map(k, j, l) // expected-note {{used here}}
#pragma omp target data map(k[:4]) // expected-error {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l) map(l[:5]) // expected-error {{variable already marked as mapped in current construct}} expected-note {{used here}}
  foo();
#pragma omp target data map(k[:4], j, l[:5]) // expected-note {{used here}}
#pragma omp target data map(k) // expected-error {{pointer cannot be mapped along with a section derived from itself}}
#pragma omp target data map(j)
#pragma omp target map(l)
  foo();

#pragma omp target data map(always, tofrom: x)
#pragma omp target data map(always: x) // expected-error {{missing map type}}
#pragma omp target data map(tofrom, always: x) // expected-error {{incorrect map type modifier, expected 'always'}} expected-error {{incorrect map type, expected one of 'to', 'from', 'tofrom', 'alloc', 'release', or 'delete'}}
#pragma omp target data map(always, tofrom: always, tofrom, x)
#pragma omp target map(tofrom j) // expected-error {{expected ',' or ')' in 'map' clause}}
  foo();
#pragma omp target private(j) map(j) // expected-error {{private variable cannot be in a map clause in '#pragma omp target' directive}}  expected-note {{defined as private}}
  {}
#pragma omp target firstprivate(j) map(j)  // expected-error {{firstprivate variable cannot be in a map clause in '#pragma omp target' directive}} expected-note {{defined as firstprivate}}
  {}
#pragma omp target map(m) // expected-error {{type 'S6<int>' is not mappable to target}}
  {}
  return tmain<int, 3>(argc)+tmain<from, 4>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 3>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<int, 4>' requested here}}
}
#endif
