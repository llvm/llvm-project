// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 -o - %s

void foo() {
}

#pragma omp task // expected-error {{unexpected OpenMP directive '#pragma omp task'}}

class S {
  S(const S &s) { a = s.a + 12; } // expected-note 14 {{implicitly declared private here}}
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
// expected-note@+1 2 {{predetermined as a firstprivate in a task construct here}}
  ++s1;
#pragma omp task default(none)
#pragma omp task default(shared)
  ++a; // expected-error 2 {{variable 'a' must have explicitly specified data sharing attributes}}
#pragma omp task default(none)
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++a; // expected-error 2 {{variable 'a' must have explicitly specified data sharing attributes}}
#pragma omp task
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++a; // expected-error {{calling a private constructor of class 'S'}}
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
  return a + b;
}

int main(int argc, char **argv) {
  int a;
  int &b = a;
  S sa;
  S &sb = sa;
  int r;
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
#pragma omp task default(none)
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

#pragma omp task default(none)
#pragma omp task default(shared)
  ++a; // expected-error {{variable 'a' must have explicitly specified data sharing attributes}}
#pragma omp task default(none)
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
#pragma omp task default(none)
#pragma omp task default(shared)
  ++sa; // expected-error {{variable 'sa' must have explicitly specified data sharing attributes}}
#pragma omp task default(none)
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++sa; // expected-error {{variable 'sa' must have explicitly specified data sharing attributes}}
#pragma omp task
#pragma omp task
  // expected-error@+1 {{calling a private constructor of class 'S'}}
  ++sa; // expected-error {{calling a private constructor of class 'S'}}
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
  // expected-note@+2 {{in instantiation of function template specialization 'foo<int>' requested here}}
  // expected-note@+1 {{in instantiation of function template specialization 'foo<S>' requested here}}
  return foo<int>() + foo<S>();
}

