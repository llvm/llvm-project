// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCFG -analyzer-config cfg-expand-default-aggr-inits=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

static char a[] = "foobar";

struct StringRef {
  const char *member = nullptr;
  int len = 3;
};

int main() {
  StringRef s{a};
  (void)s;
}

// CHECK: [B1]
// CHECK-NEXT:   1: a
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, ArrayToPointerDecay, char *)
// CHECK-NEXT:   3: [B1.2] (ImplicitCastExpr, NoOp, const char *)
// CHECK-NEXT:   4: 3
// CHECK-NEXT:   5: 
// CHECK-NEXT:   6: {[B1.1]}
// CHECK-NEXT:   7: StringRef s{a};
// CHECK-NEXT:   8: s
// CHECK-NEXT:   9: (void)[B1.8] (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

