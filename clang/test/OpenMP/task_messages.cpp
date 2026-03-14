// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-version=45 -fopenmp -ferror-limit 200 -std=c++11 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp-version=50 -fopenmp -ferror-limit 200 -std=c++11 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-version=45 -fopenmp-simd -ferror-limit 200 -std=c++11 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp-version=50 -fopenmp-simd -ferror-limit 200 -std=c++11 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp5 -DOMP51 -fopenmp -ferror-limit 100 -std=c++11 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -DOMP51 -fopenmp-simd -ferror-limit 100 -std=c++11 -o - %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp task
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void foo() {
#pragma omp task detach(0) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{'omp_event_handle_t' type not found; include <omp.h>}}
  ;
}

typedef unsigned long omp_event_handle_t;
namespace {
static int y = 0;
}
static int x = 0;

#pragma omp task // expected-error {{unexpected OpenMP directive '#pragma omp task'}}

class S {
  S(const S &s) { a = s.a + 12; } // expected-note 16 {{implicitly declared private here}}
  int a;

public:
  S() : a(0) {}
  S(int a) : a(a) {}
  operator int() { return a; }
  S &operator++() { return *this; }
  S operator+(const S &) { return *this; }
};

class S1 {
  int a;

public:
  S1() : a(0) {}
  S1 &operator++() { return *this; }
  S1(const S1 &) = delete; // expected-note 2 {{'S1' has been explicitly marked deleted here}}
};

template <class T>
int foo() {
  T a;
  T &b = a;
  int r;
  S1 s1;
// expected-error@+1 2 {{call to deleted constructor of 'S1'}}
#pragma omp task
  ++s1;
#pragma omp task default(none) // expected-note 2 {{explicit data sharing attribute requested here}}
#pragma omp task default(shared)
  ++a; // expected-error 2 {{variable 'a' must have explicitly specified data sharing attributes}}
#ifdef OMP51
#pragma omp task default(firstprivate) // omp5-note 4 {{explicit data sharing attribute requested here}}
#pragma omp task
  {
    ++x; // omp5-error 2 {{variable 'x' must have explicitly specified data sharing attributes}}
    ++y; // omp5-error 2 {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#endif

#pragma omp task default(none) // expected-note 2 {{explicit data sharing attribute requested here}}
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++a; // expected-error 2 {{variable 'a' must have explicitly specified data sharing attributes}}
#pragma omp task
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++a; // expected-error {{calling a private constructor of class 'S'}}
#pragma omp task default(shared)
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++a;
#pragma omp parallel shared(a)
#pragma omp task
#pragma omp task
  ++a;
#pragma omp parallel shared(a)
#pragma omp task default(shared)
#pragma omp task
  ++a;
#pragma omp task
#pragma omp parallel
  ++a; // expected-error {{calling a private constructor of class 'S'}}
// expected-error@+2 {{calling a private constructor of class 'S'}}
#pragma omp task
  ++b;
#pragma omp task
// expected-error@+1 2 {{calling a private constructor of class 'S'}}
#pragma omp parallel shared(a, b)
  ++a, ++b;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
// expected-error@+1 2 {{argument of a reduction clause of a parallel construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
  ++r;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task default(shared)
  // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task
  // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
  ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
// expected-error@+1 2 {{argument of a reduction clause of a for construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
    ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task default(shared)
    // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task
    // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
    ++r;
// expected-note@+1 {{non-shared variable in a task construct is predetermined as firstprivate}}
#pragma omp task
// expected-error@+2 {{reduction variable must be shared}}
// expected-error@+1 {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
#pragma omp for reduction(+ : r)
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'untied' clause}}
#pragma omp task untied untied
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'mergeable' clause}}
#pragma omp task mergeable mergeable
  ++r;
  volatile omp_event_handle_t evt;
  const omp_event_handle_t cevt = 0;
  omp_event_handle_t sevt;
  omp_event_handle_t &revt = sevt;
#pragma omp task detach // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{expected '(' after 'detach'}}
#pragma omp task detach( // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task detach() // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{expected expression}}
#pragma omp task detach(a) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'int'}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'S'}}
  ;
#pragma omp task detach(evt) detach(evt) // omp45-error 2 {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{directive '#pragma omp task' cannot contain more than one 'detach' clause}}
#pragma omp task detach(cevt) detach(revt) // omp45-error 2 {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{directive '#pragma omp task' cannot contain more than one 'detach' clause}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'const omp_event_handle_t' (aka 'const unsigned long')}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'omp_event_handle_t &' (aka 'unsigned long &')}}
#pragma omp task detach(evt) mergeable // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{'mergeable' and 'detach' clause are mutually exclusive and may not appear on the same directive}} omp5-note {{'detach' clause is specified here}}
  ;
#pragma omp task mergeable detach(evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{'detach' and 'mergeable' clause are mutually exclusive and may not appear on the same directive}} omp5-note {{'mergeable' clause is specified here}}
#pragma omp task detach(-evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{expected variable of the 'omp_event_handle_t' type}}
  ;
#pragma omp task detach(evt) shared(evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}}
#pragma omp task detach(evt) firstprivate(evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}}
  ;
  return a + b;
}

int main(int argc, char **argv) {
  int a;
  int &b = a;
  S sa;
  S &sb = sa;
  int r; // expected-note {{declared here}}
#pragma omp task { // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task( // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task[ // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task] // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task } // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task
// expected-warning@+1 {{extra tokens at the end of '#pragma omp task' are ignored}}
#pragma omp task unknown()
  foo();
L1:
  foo();
#pragma omp task
  ;
#pragma omp task
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch (argc) {
    case (0):
#pragma omp task
    {
      foo();
      break;    // expected-error {{'break' statement not in loop or switch statement}}
      continue; // expected-error {{'continue' statement not in loop statement}}
    }
    default:
      break;
    }
  }
#pragma omp task default(none) // expected-note {{explicit data sharing attribute requested here}}
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  goto L2; // expected-error {{use of undeclared label 'L2'}}
#pragma omp task
L2:
  foo();
#pragma omp task
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
#pragma omp task
      for (int n = 0; n < 100; ++n) {
  }

#pragma omp task default(none) // expected-note {{explicit data sharing attribute requested here}}
#pragma omp task default(shared)
  ++a; // expected-error {{variable 'a' must have explicitly specified data sharing attributes}}
#pragma omp task default(none) // expected-note {{explicit data sharing attribute requested here}}
#pragma omp task
  ++a; // expected-error {{variable 'a' must have explicitly specified data sharing attributes}}
#pragma omp task default(shared)
#pragma omp task
  ++a;
#pragma omp task
#pragma omp parallel
  ++a;
#pragma omp task
  ++b;
#pragma omp task
#pragma omp parallel shared(a, b)
  ++a, ++b;
#pragma omp task default(none) // expected-note {{explicit data sharing attribute requested here}}
#pragma omp task default(shared)
  ++sa; // expected-error {{variable 'sa' must have explicitly specified data sharing attributes}}
#pragma omp task default(none) // expected-note {{explicit data sharing attribute requested here}}
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++sa; // expected-error {{variable 'sa' must have explicitly specified data sharing attributes}}
#pragma omp task
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++sa; // expected-error {{calling a private constructor of class 'S'}}
#pragma omp task default(shared)
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++sa;
#pragma omp parallel shared(sa)
#pragma omp task
#pragma omp task
  ++sa;
#pragma omp parallel shared(sa)
#pragma omp task default(shared)
#pragma omp task
  ++sa;
#pragma omp task
#pragma omp parallel
  ++sa; // expected-error {{calling a private constructor of class 'S'}}
// expected-error@+2 {{calling a private constructor of class 'S'}}
#pragma omp task
  ++sb;
// expected-error@+2 2 {{calling a private constructor of class 'S'}}
#pragma omp task
#pragma omp parallel shared(sa, sb)
  ++sa, ++sb;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
// expected-error@+1 {{argument of a reduction clause of a parallel construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
  // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task default(shared)
  // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task
  // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
  ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
// expected-error@+1 {{argument of a reduction clause of a for construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
    // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task default(shared)
    // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task
    // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
    ++r;
// expected-note@+1 {{non-shared variable in a task construct is predetermined as firstprivate}}
#pragma omp task
// expected-error@+2 {{reduction variable must be shared}}
// expected-error@+1 {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
#pragma omp for reduction(+ : r)
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'untied' clause}}
#pragma omp task untied untied
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'mergeable' clause}}
#pragma omp task mergeable mergeable
  ++r;
// expected-error@+4 {{variable length arrays are not supported in OpenMP tasking regions with 'untied' clause}}
// expected-note@+3 {{read of non-const variable 'r' is not allowed in a constant expression}}
#pragma omp task untied
  {
    int array[r];
  }
  volatile omp_event_handle_t evt;
  omp_event_handle_t sevt;
  const omp_event_handle_t cevt = evt;
  omp_event_handle_t &revt = sevt;
#pragma omp task detach // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{expected '(' after 'detach'}}
#pragma omp task detach( // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task detach() // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{expected expression}}
#pragma omp task detach(a) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'int'}}
#pragma omp task detach(evt) detach(evt) // omp45-error 2 {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{directive '#pragma omp task' cannot contain more than one 'detach' clause}}
#pragma omp task detach(cevt) detach(revt) // omp45-error 2 {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} expected-error {{directive '#pragma omp task' cannot contain more than one 'detach' clause}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'const omp_event_handle_t' (aka 'const unsigned long')}} omp5-error {{expected variable of the 'omp_event_handle_t' type, not 'omp_event_handle_t &' (aka 'unsigned long &')}}
#pragma omp task detach(evt) mergeable // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{'mergeable' and 'detach' clause are mutually exclusive and may not appear on the same directive}} omp5-note {{'detach' clause is specified here}}
  ;
#pragma omp task mergeable detach(evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{'detach' and 'mergeable' clause are mutually exclusive and may not appear on the same directive}} omp5-note {{'mergeable' clause is specified here}}
#pragma omp task detach(-evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}} omp5-error {{expected variable of the 'omp_event_handle_t' type}}
  ;
#pragma omp task detach(evt) shared(evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}}
#pragma omp task detach(evt) firstprivate(evt) // omp45-error {{unexpected OpenMP clause 'detach' in directive '#pragma omp task'}}
  ;
  // expected-note@+2 {{in instantiation of function template specialization 'foo<int>' requested here}}
  // expected-note@+1 {{in instantiation of function template specialization 'foo<S>' requested here}}
  return foo<int>() + foo<S>();
}

