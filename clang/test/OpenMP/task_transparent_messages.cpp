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

void TestTaskTransparentInvalidArgs() {
  #pragma omp task transparent(invalid_arg) // expected-error {{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
  #pragma omp task transparent(123) // expected-error {{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
#pragma omp task transparent(omp_import, omp_not_import) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task transparent() // expected-error {{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
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
  #pragma omp taskloop transparent(invalid_arg) // expected-error {{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(123) // expected-error {{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
  for (int i = 0; i < 10; ++i) {}
#pragma omp taskloop transparent(omp_not_import, omp_import) // expected-error{{expected ')'}} // expected-note{{to match this '('}}  // expected-error{{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent() // expected-error {{expected 'omp_not_impex', 'omp_import', 'omp_export' or 'omp_impex' in OpenMP clause 'transparent'}}
  for (int i = 0; i < 10; ++i) {}
}

#else
void TransparentClauseNotSupported() {
  #pragma omp task transparent(omp_pool) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}}
  #pragma omp task transparent(omp_team) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp task'}}
  #pragma omp taskloop transparent(omp_team) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 10; ++i) {}
  #pragma omp taskloop transparent(omp_pool) // omp45-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp50-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}} omp51-error {{unexpected OpenMP clause 'transparent' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 10; ++i) {}
}
#endif
