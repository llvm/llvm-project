// RUN: %clang_cc1 -std=c++11 -fsyntax-only -analyze -analyzer-checker=debug.DumpCFG %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

struct X { X(const X&); };
void f(X x) { (void) [x]{}; }

// CHECK: [B2 (ENTRY)]
// CHECK:   Succs (1): B1
// CHECK: [B1]
// CHECK:   1: x
// CHECK:   2: [B1.1] (ImplicitCastExpr, NoOp, const struct X)
// CHECK:   3: [B1.2] (CXXConstructExpr, struct X)
// CHECK:   4: [=x]     {
// CHECK:    }
// CHECK:   5: (void)[B1.4] (CStyleCastExpr, ToVoid, void)
// CHECK:   Preds (1): B2
// CHECK:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK:   Preds (1): B1

