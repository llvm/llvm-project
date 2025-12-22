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
  int a;  // expected-note{{declared here}} // expected-note{{declared here}} // expected-note{{declared here}}
  // expected-error@+1{{expression is not an integral constant expression}} // expected-note@+1{{read of non-const variable 'a' is not allowed in a constant expression}}
#pragma omp task transparent(a)

#pragma omp parallel
  {
    // expected-error@+1{{expression is not an integral constant expression}} // expected-note@+1{{read of non-const variable 'a' is not allowed in a constant expression}}
#pragma omp task transparent(a)
    {
      // expected-error@+1{{expression is not an integral constant expression}} // expected-note@+1 {{read of non-const variable 'a' is not allowed in a constant expression}}
#pragma omp taskloop transparent(a)
      for (int i = 0; i < 5; ++i) {}
    }
  }
  // expected-error@+1{{use of undeclared identifier 'omp_not_impex'}}
#pragma omp task transparent(omp_not_impex)
  // expected-error@+1{{use of undeclared identifier 'omp_import'}}
#pragma omp task transparent(omp_import)
  // expected-error@+1{{use of undeclared identifier 'omp_export'}}
#pragma omp task transparent(omp_export)
  // expected-error@+1{{use of undeclared identifier 'omp_impex'}}
#pragma omp task transparent(omp_impex)
  for (int i = 0; i < 5; ++i) {}

}

template <int C>
void TestTaskLoopImpex() {
  // expected-error@+1{{invalid value for transparent clause, expected one of: omp_not_impex, omp_import, omp_export, omp_impex}}
#pragma omp taskloop transparent(C)
  for (int i = 0; i < 10; ++i) {}
}

void run() {
  // expected-note@+1{{in instantiation of function template specialization 'TestTaskLoopImpex<4>' requested here}}
  TestTaskLoopImpex<4>();
}

typedef void **omp_impex_t;
extern const omp_impex_t omp_not_impex; // omp60-note {{'omp_not_impex' declared here}}
extern const omp_impex_t omp_import;
extern const omp_impex_t omp_export;
extern const omp_impex_t omp_impex;

template <typename T>
class TransparentTemplate {
public:
  void TestTaskImport() {
    #pragma omp task transparent(omp_import)
    {
      T temp;
    }
  }
  void TestTaskLoopImpex() {
    #pragma omp taskloop transparent(omp_impex)
    for (int i = 0; i < 10; ++i) {}
  }
};

void TestTaskTransparent() {
  int a;
#pragma omp task transparent(omp_not_impex)
#pragma omp task transparent(omp_import)
#pragma omp task transparent(omp_export)
#pragma omp task transparent(omp_impex)

#pragma omp task transparent(omp_import) if(1)
#pragma omp task transparent(omp_impex) priority(5)
#pragma omp task transparent(omp_not_impex) depend(out: a)
#pragma omp parallel
  {
#pragma omp task transparent(omp_export)
    {
#pragma omp taskloop transparent(omp_not_impex)
      for (int i = 0; i < 5; ++i) {}
    }
  }

  TransparentTemplate<int> obj;
  obj.TestTaskImport();
  obj.TestTaskLoopImpex();
}

int invalid_arg; //expected-note{{declared here}} //expected-note{{declared here}}
void TestTaskTransparentInvalidArgs() {
  #pragma omp task transparent(invalid_arg) // expected-error{{expression is not an integral constant expression}} // expected-note{{read of non-const variable 'invalid_arg' is not allowed in a constant expression}}
  #pragma omp task transparent(123) // expected-error{{invalid value for transparent clause, expected one of: omp_not_impex, omp_import, omp_export, omp_impex}}
#pragma omp task transparent(omp_import, omp_not_import) // expected-error{{expected ')'}} // expected-note{{to match this '('}}
  #pragma omp task transparent() // expected-error {{expected expression}}
  {}
}

void TestTaskloopTransparent() {
  #pragma omp taskloop transparent(omp_import)
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(omp_export)
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(omp_not_impex) grainsize(5)
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(omp_impex) num_tasks(2)
  for (int i = 0; i < 10; ++i) {}
}


void TestTaskLoopTransparentInvalidArgs() {
  #pragma omp taskloop transparent(invalid_arg) // expected-error {{expression is not an integral constant expression}} // expected-note {{read of non-const variable 'invalid_arg' is not allowed in a constant expression}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(123) // expected-error {{invalid value for transparent clause, expected one of: omp_not_impex, omp_import, omp_export, omp_impex}}
  for (int i = 0; i < 10; ++i) {}
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
