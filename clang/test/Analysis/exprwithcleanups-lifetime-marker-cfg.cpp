// RUN: %clang_analyze_cc1 -fcxx-exceptions -fexceptions -analyzer-checker=debug.DumpCFG -analyzer-config cfg-lifetime=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

struct TrivialDtor {};

struct NonTrivialDtor {
    ~NonTrivialDtor();
};

void foo(const TrivialDtor&, const NonTrivialDtor&);

void bar(const TrivialDtor& = TrivialDtor());

// CHECK:      4: [B1.3] (ImplicitCastExpr, NoOp, const TrivialDtor)
// CHECK-NEXT: 5: [B1.4]
// CHECK:      8: [B1.7] (ImplicitCastExpr, NoOp, const NonTrivialDtor)
// CHECK-NEXT: 9: [B1.8]
// CHECK:      (FullExprCleanup collected 2 MTEs: [B1.4], [B1.8])
void f() {
  foo(TrivialDtor(), NonTrivialDtor());
}

// CHECK: (FullExprCleanup collected 1 MTE: TrivialDtor())
void g() {
  bar();
}
