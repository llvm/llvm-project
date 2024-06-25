// RUN: %clang_cc1 -verify=expected,lt50,lt51 -fopenmp -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,lt51 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,ge51 -fopenmp -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,lt50,lt51 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,lt51 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,ge51 -fopenmp-simd -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,ge50,ge51,cxx23 -fopenmp -fopenmp-simd -x c++ -std=c++23 %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp target update to(x)
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

static int y;
#pragma omp declare target(y)

void yyy() {
#pragma omp target update to(y) // expected-error {{the host cannot update a declare target variable that is not externally visible}}
}

int __attribute__((visibility("hidden"))) z;
#pragma omp declare target(z)

void zzz() {
#pragma omp target update from(z) // expected-error {{the host cannot update a declare target variable that is not externally visible}}
}

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // Aexpected-note {{declared here}}

template <class T, class S> // Aexpected-note {{declared here}}
int tmain(T argc, S **argv) {
  int n;
  return 0;
}

struct S {
  int i;
};

int main(int argc, char **argv) {
  int m;
  #pragma omp target update // expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(m) { // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ( // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) [ // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ] // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}

  #pragma omp declare mapper(id: S s) map(s.i)
  S s;

  // Check parsing with no modifiers.
  // lt51-error@+2 {{expected expression}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(: s)
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(:)
  // expected-error@+2 2 {{expected expression}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(,:)

  // Check parsing with one modifier.
  // expected-error@+2 {{use of undeclared identifier 'foobar'}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(foobar: s)
  // expected-error@+3 {{expected ',' or ')' in 'to' clause}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp target update to(m: s)
  #pragma omp target update to(mapper(id): s)
  // lt51-error@+2 {{use of undeclared identifier 'present'}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present: s)
  // ge51-warning@+4 {{missing ':' after motion modifier - ignoring}}
  // lt51-warning@+3 {{missing ':' after ) - ignoring}}
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(mapper(id) s)
  // ge51-warning@+4 {{missing ':' after motion modifier - ignoring}}
  // ge51-error@+3 {{expected expression}}
  // lt51-error@+2 {{use of undeclared identifier 'present'}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present s)
  // ge51-warning@+4 {{missing ':' after motion modifier - ignoring}}
  // lt51-warning@+3 {{missing ':' after ) - ignoring}}
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(mapper(id))
  // ge51-warning@+4 {{missing ':' after motion modifier - ignoring}}
  // ge51-error@+3 {{expected expression}}
  // lt51-error@+2 {{use of undeclared identifier 'present'}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present)
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(mapper(id):)
  // ge51-error@+3 {{expected expression}}
  // lt51-error@+2 {{use of undeclared identifier 'present'}}
  // expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present:)

  // Check parsing with two modifiers.
  // lt51-warning@+1 {{missing ':' after ) - ignoring}}
  #pragma omp target update to(mapper(id), present: s)
  // lt51-error@+3 {{use of undeclared identifier 'present'}}
  // lt51-error@+2 {{use of undeclared identifier 'id'}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present, mapper(id): s)
  // lt51-warning@+1 {{missing ':' after ) - ignoring}}
  #pragma omp target update to(mapper(id) present: s)
  // lt51-error@+2 {{use of undeclared identifier 'present'}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present mapper(id): s)

  // Check parsing with unnecessary commas.
  // lt51-warning@+1 {{missing ':' after ) - ignoring}}
  #pragma omp target update to(mapper(id),: s)
  // lt51-error@+3 {{use of undeclared identifier 'present'}}
  // lt51-error@+2 {{expected expression}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present , : s)
  // ge51-warning@+2 {{missing ':' after motion modifier - ignoring}}
  // lt51-warning@+1 {{missing ':' after ) - ignoring}}
  #pragma omp target update to(mapper(id),,: s)
  // ge51-warning@+5 {{missing ':' after motion modifier - ignoring}}
  // lt51-error@+4 {{use of undeclared identifier 'present'}}
  // lt51-error@+3 {{expected expression}}
  // lt51-error@+2 {{expected expression}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present,,: s)
  // lt51-warning@+1 {{missing ':' after ) - ignoring}}
  #pragma omp target update to(mapper(id), present,: s)
  // lt51-error@+4 {{use of undeclared identifier 'present'}}
  // lt51-error@+3 {{use of undeclared identifier 'id'}}
  // lt51-error@+2 {{expected expression}}
  // lt51-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(present, mapper(id),: s)

  #pragma omp target update from(m) allocate(m) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp target update'}}
  {
    foo();
  }

  double marr[10][5][10];
#pragma omp target update to(marr[0:2][2:4][1:2]) // lt50-error {{array section does not specify contiguous storage}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
#pragma omp target update from(marr[0:2][2:4][1:2]) // lt50-error {{array section does not specify contiguous storage}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

#pragma omp target update to(marr[0:][1:2:2][1:2]) // ge50-error {{array section does not specify length for outermost dimension}} lt50-error {{expected ']'}} lt50-note {{to match this '['}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
#pragma omp target update from(marr[0:][1:2:2][1:2]) // ge50-error {{array section does not specify length for outermost dimension}} lt50-error {{expected ']'}} lt50-note {{to match this '['}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

  int arr[4][3][2][1];
#pragma omp target update to(arr[0:2][2:4][:2][1]) // lt50-error {{array section does not specify contiguous storage}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
#pragma omp target update from(arr[0:2][2:4][:2][1]) // lt50-error {{array section does not specify contiguous storage}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

  double ***dptr;
#pragma omp target update to(dptr[0:2][2:4][1:2]) // lt50-error {{array section does not specify contiguous storage}} ge50-error 2 {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
#pragma omp target update from(dptr[0:2][2:4][1:2]) // lt50-error {{array section does not specify contiguous storage}} ge50-error 2 {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}} lt50-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

  int iarr[5][5];
// ge50-error@+4 {{section stride is evaluated to a non-positive value -1}}
// lt50-error@+3 {{expected ']'}}
// lt50-note@+2 {{to match this '['}}
// expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(iarr[0:][1:2:-1])
  {}
// ge50-error@+4 {{section stride is evaluated to a non-positive value -1}}
// lt50-error@+3 {{expected ']'}}
// lt50-note@+2 {{to match this '['}}
// expected-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(iarr[0:][1:2:-1])
  {}
// lt50-error@+5 {{expected expression}}
// ge50-error@+4 {{array section does not specify length for outermost dimension}}
// lt50-error@+3 {{expected ']'}}
// lt50-note@+2 {{to match this '['}}
// lt50-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(iarr[0: :2][1:2])
  {}
// lt50-error@+5 {{expected expression}}
// ge50-error@+4 {{array section does not specify length for outermost dimension}}
// lt50-error@+3 {{expected ']'}}
// lt50-note@+2 {{to match this '['}}
// lt50-error@+1 {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(iarr[0: :2][1:2])
  {}

  return tmain(argc, argv);
}

template<typename _Tp, int _Nm> struct array {
  _Tp & operator[](int __n) noexcept;
};

#pragma omp declare target
extern array<double, 4> arr;
#pragma omp end declare target

void copy_host_to_device()
{
  #pragma omp target update from(arr)  // expected-no-error
  arr[0] = 0;
}

struct FOO; // expected-note {{forward declaration of 'FOO'}}
extern FOO a;
template <typename T, int I>
struct bar {
  void func() {
    #pragma omp target map(to: a) // expected-error {{incomplete type 'FOO' where a complete type is required}}
    foo();
  }
};

#if defined(__cplusplus) && __cplusplus >= 202101L

namespace cxx23 {

struct S {
  int operator[](auto...);
};

void f() {

  int test[10];

#pragma omp target update to(test[1])

#pragma omp target update to(test[1, 2]) // cxx23-error {{type 'int[10]' does not provide a subscript operator}} \
                                         // cxx23-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

#pragma omp target update to(test [1:1:1])

#pragma omp target update to(test [1, 2:1:1]) // cxx23-error {{expected ']'}} // expected-note {{'['}} \
                                            // cxx23-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

#pragma omp target update to(test [1, 2:]) // cxx23-error {{expected ']'}} // expected-note {{'['}} \
                                            // cxx23-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

#pragma omp target update to(test[1, 2 ::]) // cxx23-error {{expected ']'}} // expected-note {{'['}} \
                                            // cxx23-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

#pragma omp target update to(test[]) // cxx23-error {{type 'int[10]' does not provide a subscript operator}} \
                                            // cxx23-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  S s;
  (void)s[0];
  (void)s[];
  (void)s[1, 2];
}

}

#endif
