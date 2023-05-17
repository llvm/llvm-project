// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++11 -w %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX11,CXX11-ELIDE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++17 -w %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX17 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++11 -w -analyzer-config elide-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX11,CXX11-NOELIDE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++17 -w -analyzer-config elide-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX17 %s

class D {
public:
  D();
  ~D();
};

@interface E {}
-(void) foo: (D) d;
-(D) bar;
@end

// CHECK: void passArgumentIntoMessage(E *e)
// CHECK:          1: e
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, LValueToRValue, E *)
// CXX11-ELIDE-NEXT:     3: D() (CXXConstructExpr, [B1.4], [B1.6], [B1.7], D)
// CXX11-NOELIDE-NEXT:     3: D() (CXXConstructExpr, [B1.4], [B1.6], D)
// CXX11-NEXT:     4: [B1.3] (BindTemporary)
// CXX11-NEXT:     5: [B1.4] (ImplicitCastExpr, NoOp, const D)
// CXX11-NEXT:     6: [B1.5]
// CXX11-NEXT:     7: [B1.6] (CXXConstructExpr, [B1.8], [B1.9]+0, D)
// CXX11-NEXT:     8: [B1.7] (BindTemporary)
// Double brackets trigger FileCheck variables, escape.
// CXX11-NEXT:     9: {{\[}}[B1.2] foo:[B1.8]]
// CXX11-NEXT:    10: ~D() (Temporary object destructor)
// CXX11-NEXT:    11: ~D() (Temporary object destructor)
// CXX17-NEXT:     3: D() (CXXConstructExpr, [B1.4], [B1.5]+0, D)
// CXX17-NEXT:     4: [B1.3] (BindTemporary)
// Double brackets trigger FileCheck variables, escape.
// CXX17-NEXT:     5: {{\[}}[B1.2] foo:[B1.4]]
// CXX17-NEXT:     6: ~D() (Temporary object destructor)
void passArgumentIntoMessage(E *e) {
  [e foo: D()];
}

// CHECK: void returnObjectFromMessage(E *e)
// CHECK:        [B1]
// CHECK-NEXT:     1: e
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, LValueToRValue, E *)
// Double brackets trigger FileCheck variables, escape.
// CXX11-ELIDE-NEXT:     3: {{\[}}[B1.2] bar] (CXXRecordTypedCall, [B1.4], [B1.6], [B1.7])
// CXX11-NOELIDE-NEXT:     3: {{\[}}[B1.2] bar] (CXXRecordTypedCall, [B1.4], [B1.6])
// CXX11-NEXT:     4: [B1.3] (BindTemporary)
// CXX11-NEXT:     5: [B1.4] (ImplicitCastExpr, NoOp, const D)
// CXX11-NEXT:     6: [B1.5]
// CXX11-NEXT:     7: [B1.6] (CXXConstructExpr, [B1.8], D)
// CXX11-NEXT:     8: D d = [e bar];
// CXX11-NEXT:     9: ~D() (Temporary object destructor)
// CXX11-NEXT:    10: [B1.8].~D() (Implicit destructor)
// Double brackets trigger FileCheck variables, escape.
// CXX17-NEXT:     3: {{\[}}[B1.2] bar] (CXXRecordTypedCall, [B1.5], [B1.4])
// CXX17-NEXT:     4: [B1.3] (BindTemporary)
// CXX17-NEXT:     5: D d = [e bar];
// CXX17-NEXT:     6: [B1.5].~D() (Implicit destructor)
void returnObjectFromMessage(E *e) {
  D d = [e bar];
}
