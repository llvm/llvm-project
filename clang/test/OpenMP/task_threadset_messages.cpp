// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected -DOMP60 -fopenmp -fopenmp-version=60 -std=c++11 -ferror-limit 200 -o - %s

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -fopenmp-version=51 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected -DOMP60 -fopenmp-simd -fopenmp-version=60 -std=c++11 -ferror-limit 200 -o - %s

#ifdef OMP60
struct ComplexStruct {
  int data[10];
  struct InnerStruct {
    float value;
  } inner;
};

// Template class with member functions using 'threadset'.
template <typename T>
class TemplateClass {
public:
  void foo() {
    #pragma omp task threadset(omp_pool)
    {
      T temp;
    }
  }
  void bar() {
    #pragma omp taskloop threadset(omp_team)
    for (int i = 0; i < 10; ++i) {}
  }
};

// Valid uses of 'threadset' with 'omp_pool' and 'omp_team' in task directive.
void test_task_threadset_valid() {
  int a;
  #pragma omp task threadset(omp_pool)
  #pragma omp task threadset(omp_team)
  #pragma omp task threadset(omp_pool) if(1)
  #pragma omp task threadset(omp_team) priority(5)
  #pragma omp task threadset(omp_pool) depend(out: a)
  #pragma omp parallel
  {
    #pragma omp task threadset(omp_pool)
    {
      #pragma omp taskloop threadset(omp_team)
      for (int i = 0; i < 5; ++i) {}
    }
  }

  TemplateClass<int> obj;
  obj.foo();
  obj.bar();
}

// Invalid uses of 'threadset' with incorrect arguments in task directive.
void test_task_threadset_invalid_args() {
  #pragma omp task threadset(invalid_arg) // expected-error {{expected 'omp_pool' or 'omp_team' in OpenMP clause 'threadset'}}
  #pragma omp task threadset(123) // expected-error {{expected 'omp_pool' or 'omp_team' in OpenMP clause 'threadset'}}
  #pragma omp task threadset(omp_pool, omp_team) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task threadset() // expected-error {{expected 'omp_pool' or 'omp_team' in OpenMP clause 'threadset'}}
  {}
}

// Valid uses of 'threadset' with 'omp_pool' and 'omp_team' in taskloop directive.
void test_taskloop_threadset_valid() {
  #pragma omp taskloop threadset(omp_pool)
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset(omp_team)
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset(omp_pool) grainsize(5)
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset(omp_team) num_tasks(2)
  for (int i = 0; i < 10; ++i) {}
}

// Invalid uses of 'threadset' with incorrect arguments in taskloop directive.
void test_taskloop_threadset_invalid_args() {
  #pragma omp taskloop threadset(invalid_arg) // expected-error {{expected 'omp_pool' or 'omp_team' in OpenMP clause 'threadset'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset(123) // expected-error {{expected 'omp_pool' or 'omp_team' in OpenMP clause 'threadset'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset(omp_pool, omp_team) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset() // expected-error {{expected 'omp_pool' or 'omp_team' in OpenMP clause 'threadset'}}
  for (int i = 0; i < 10; ++i) {}
}

#else
void test_threadset_not_supported() {
  #pragma omp task threadset(omp_pool) // omp45-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp task'}} omp50-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp task'}} omp51-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp task'}}
  #pragma omp task threadset(omp_team) // omp45-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp task'}} omp50-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp task'}} omp51-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp task'}}
  #pragma omp taskloop threadset(omp_team) // omp45-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp taskloop'}} omp50-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp taskloop'}} omp51-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop threadset(omp_pool) // omp45-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp taskloop'}} omp50-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp taskloop'}} omp51-error {{unexpected OpenMP clause 'threadset' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 10; ++i) {}
}
#endif
