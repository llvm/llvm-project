// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp60 -DOMP60 -fopenmp -fopenmp-version=60 -ferror-limit 200 -o - %s

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -fopenmp-version=51 -std=c++11 -ferror-limit 200 -o - %s
// RUN: %clang_cc1 -verify=expected,omp60 -DOMP60 -fopenmp-simd -fopenmp-version=60 -std=c++11 -ferror-limit 200 -o - %s

#ifdef OMP60
struct ComplexStruct {
  int data[10];
  struct InnerStruct {
    float value;
  } inner;
};

void TestTaskTransparentWithErrors() {
  int x = 10;
  int* ptr = &x;
  int arr[5];
#pragma omp task transparent() // expected-error{{expected expression}}
  // expected-error@+1{{use of undeclared identifier 'omp_not_impex'}}
#pragma omp task transparent(omp_not_impex)
  // expected-error@+1{{use of undeclared identifier 'omp_import'}}
#pragma omp task transparent(omp_import)
  // expected-error@+1{{use of undeclared identifier 'omp_export'}}
#pragma omp task transparent(omp_export)
  // expected-error@+1{{use of undeclared identifier 'omp_impex'}}
#pragma omp task transparent(omp_impex)
  // expected-error@+1{{invalid value for transparent clause, expected one of: omp_not_impex, omp_import, omp_export, omp_impex}}
#pragma omp task transparent(5)
  // expected-error@+1{{transparent clause cannot be applied to type: 'int *'}}
#pragma omp task transparent(ptr)
  // expected-error@+1{{transparent clause cannot be applied to type: 'int *'}}
#pragma omp task transparent(&x)
  // expected-error@+1{{transparent clause cannot be applied to type: 'double'}}
#pragma omp task transparent(20.0)
  // expected-error@+1{{transparent clause cannot be applied to type: 'int[5]'}}
#pragma omp task transparent(arr)
  for (int i = 0; i < 5; ++i) {}
}

typedef void **omp_impex_t;
extern const omp_impex_t omp_not_impex; // omp60-note {{'omp_not_impex' declared here}}
extern const omp_impex_t omp_import;
extern const omp_impex_t omp_export;
extern const omp_impex_t omp_impex;

int invalid_arg;
void TestTaskTransparentInvalidArgs() {
#pragma omp task transparent(omp_import, omp_not_import) // expected-error{{expected ')'}} // expected-note{{to match this '('}}
  #pragma omp task transparent() // expected-error {{expected expression}}
  {}
}

void TestTaskLoopTransparentInvalidArgs() {
#pragma omp taskloop transparent(omp_not_import, omp_import) // expected-error{{expected ')'}} // expected-note{{to match this '('}}  // expected-error{{use of undeclared identifier 'omp_not_import'; did you mean 'omp_not_impex'?}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent() // expected-error {{expected expression}}
  for (int i = 0; i < 10; ++i) {}
}

#else
void TransparentClauseNotSupported() {
  #pragma omp task transparent(omp_pool) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp45-error {{use of undeclared identifier 'omp_pool'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp50-error {{use of undeclared identifier 'omp_pool'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp51-error {{use of undeclared identifier 'omp_pool'}}
  #pragma omp task transparent(omp_team) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp45-error {{use of undeclared identifier 'omp_team'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp50-error {{use of undeclared identifier 'omp_team'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp51-error {{use of undeclared identifier 'omp_team'}}
  #pragma omp taskloop transparent(omp_team) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp45-error {{use of undeclared identifier 'omp_team'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp50-error {{use of undeclared identifier 'omp_team'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp51-error {{use of undeclared identifier 'omp_team'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(omp_pool) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp45-error {{use of undeclared identifier 'omp_pool'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp50-error {{use of undeclared identifier 'omp_pool'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp51-error {{use of undeclared identifier 'omp_pool'}}
  for (int i = 0; i < 10; ++i) {}
}
#endif
