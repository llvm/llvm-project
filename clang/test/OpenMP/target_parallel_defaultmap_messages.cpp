// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 %s -verify=expected,omp51 -Wuninitialized -DOMP51
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 %s -verify=expected,omp51 -Wuninitialized -DOMP51

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 %s -verify=expected,omp5 -Wuninitialized -DOMP5
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 %s -verify=expected,omp5 -Wuninitialized -DOMP5

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 %s -verify=expected,omp45 -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 %s -verify=expected,omp45 -Wuninitialized

struct Bar {
  int a;
};

class Baz {
  public:
    Bar bar;
    int *p;
};

int *g;
int arr[50];
Bar bar;
Baz baz;
Baz* bazPtr = &baz;

void foo() {
}

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
#pragma omp target parallel defaultmap( // omp51-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default', 'present' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap() // omp51-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default', 'present' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target parallel defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} omp51-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom) // omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom, // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target parallel defaultmap (scalar: // expected-error {{expected ')'}} omp51-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} omp51-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default', 'present' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom, scalar // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
#ifdef OMP5
  #pragma omp target parallel defaultmap(tofrom:scalar) defaultmap(to:scalar) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target parallel defaultmap(alloc:pointer) defaultmap(to:scalar) defaultmap(firstprivate:pointer) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target parallel defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  g++; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:pointer) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:pointer) defaultmap(none:scalar) map(g) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:pointer) defaultmap(none:scalar) defaultmap(none:aggregate) map(g) firstprivate(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  argc = *g + arr[1]; // expected-error {{variable 'arr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  bar.a += argc; // expected-error {{variable 'bar' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  int vla[argc];
  #pragma omp target parallel defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  vla[argc-1] += argc; // expected-error {{variable 'vla' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  *baz.p += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  bazPtr->p += argc; // expected-error {{variable 'bazPtr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

#endif

  return argc;
}

int main(int argc, char **argv) {
#pragma omp target parallel defaultmap( // omp51-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default', 'present' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap() // omp51-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default', 'present' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target parallel defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} omp51-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom) // omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom, // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target parallel defaultmap (scalar: // expected-error {{expected ')'}} omp51-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default', 'present' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp51-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
#pragma omp target parallel defaultmap(tofrom, scalar // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
#ifdef OMP5
  #pragma omp target parallel defaultmap(tofrom:scalar) defaultmap(to:scalar) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target parallel defaultmap(alloc:pointer) defaultmap(to:scalar) defaultmap(firstprivate:pointer) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target parallel defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  g++; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:pointer) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:pointer) defaultmap(none:scalar) map(g) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:pointer) defaultmap(none:scalar) defaultmap(none:aggregate) map(g) firstprivate(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  argc = *g + arr[1]; // expected-error {{variable 'arr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  bar.a += argc; // expected-error {{variable 'bar' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target parallel defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  int vla[argc];
  #pragma omp target parallel defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  vla[argc-1] += argc; // expected-error {{variable 'vla' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target parallel defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  *baz.p += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_device_ptr clause requested here}}
  #pragma omp parallel
  bazPtr->p += argc; // expected-error {{variable 'bazPtr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
#endif

  return tmain<int, char, 1, 0>(argc, argv);  // omp5-note {{in instantiation of function template specialization 'tmain<int, char, 1, 0>' requested here}}
}
