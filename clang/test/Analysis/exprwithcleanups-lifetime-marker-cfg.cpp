// RUN: %clang_analyze_cc1 -fcxx-exceptions -fexceptions -analyzer-checker=debug.DumpCFG -analyzer-config cfg-lifetime=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

struct TrivialDtor {};

struct NonTrivialDtor {
    ~NonTrivialDtor();
};

void foo(const TrivialDtor&, const NonTrivialDtor&);

// CHECK: (FullExprCleanup collected 2 MTEs: [B1.4], [B1.8])
void f() {
  foo(TrivialDtor(), NonTrivialDtor());
}