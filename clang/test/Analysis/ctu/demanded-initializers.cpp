// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -emit-pch -o %t/unused.ast %t/unused.cpp
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -emit-pch -o %t/dependency.ast %t/dependency.cpp
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -emit-pch -o %t/dependent.ast %t/dependent.cpp
//
// DEFINE: %{analyze} = %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c++17 \
// DEFINE:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores \
// DEFINE:   -analyzer-config eagerly-assume=false,experimental-enable-naive-ctu-analysis=true,ctu-dir=%t,ctu-import-cpp-threshold=2,ctu-phase1-inlining=all
//
// RUN: %{analyze} -analyze-function="local_initializer()" -verify=local %t/main.cpp
// RUN: %{analyze} -analyze-function="imported_initializer()" -verify=imported %t/main.cpp
// RUN: %{analyze} -analyzer-config ctu-import-cpp-threshold=1 -analyze-function="constexpr_if()" -verify=cfg %t/main.cpp
// RUN: %{analyze} -analyzer-config ctu-import-cpp-threshold=1 -analyze-function="nested_body()" -verify=nested %t/main.cpp

//--- main.cpp
void clang_analyzer_eval(bool);
void observe(int);
extern const int unused;
extern const int dependency;
extern const int dependent;

const int local_dependent = dependency + 1;

void local_initializer() {
  clang_analyzer_eval(local_dependent == 8); // local-warning{{TRUE}}
}

void imported_initializer() {
  clang_analyzer_eval(dependent == 8); // imported-warning{{TRUE}}
}

void constexpr_if() {
  int value;
  value = 0; // cfg-warning{{Value stored to 'value' is never read}}
  if constexpr (int key = dependency; true) {
    if (dependency == 7)
      value = key;
  } else {
    observe(unused);
  }
  observe(value);
}

void nested_body() {
  // An uncalled lambda body must not consume the slot for dependency.
  auto lambda = [] { return unused; };
  (void)lambda;
  clang_analyzer_eval(dependency == 7); // nested-warning{{TRUE}}
}

//--- unused.cpp
extern const int unused = 123;

//--- dependency.cpp
extern const int dependency = 7;

//--- dependent.cpp
extern const int dependency;
extern const int dependent = dependency + 1;

//--- externalDefMap.txt
9:c:@unused unused.ast
13:c:@dependency dependency.ast
12:c:@dependent dependent.ast
