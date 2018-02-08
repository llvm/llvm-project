// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -analyzer-config cfg-temporary-dtors=true -std=c++11 %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

class C {
public:
  C();
  C(C *);
};

typedef __typeof(sizeof(int)) size_t;
void *operator new(size_t size, void *placement);

namespace operator_new {

// CHECK: void operatorNewWithConstructor()
// CHECK:          1: CFGNewAllocator(C *)
// CHECK-NEXT:     2:  (CXXConstructExpr, [B1.3], class C)
// CHECK-NEXT:     3: new C([B1.2])
void operatorNewWithConstructor() {
  new C();
}

// CHECK: void operatorNewWithConstructorWithOperatorNewWithContstructor()
// CHECK:          1: CFGNewAllocator(C *)
// CHECK-NEXT:     2: CFGNewAllocator(C *)
// CHECK-NEXT:     3:  (CXXConstructExpr, [B1.4], class C)
// CHECK-NEXT:     4: new C([B1.3])
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], class C)
// CHECK-NEXT:     6: new C([B1.5])
void operatorNewWithConstructorWithOperatorNewWithContstructor() {
	new C(new C());
}

// CHECK: void operatorPlacementNewWithConstructorWithinPlacementArgument()
// CHECK:          1: CFGNewAllocator(C *)
// CHECK-NEXT:     2:  (CXXConstructExpr, [B1.3], class C)
// CHECK-NEXT:     3: new C([B1.2])
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, BitCast, void *)
// CHECK-NEXT:     5: CFGNewAllocator(C *)
// CHECK-NEXT:     6:  (CXXConstructExpr, [B1.7], class C)
// CHECK-NEXT:     7: new ([B1.4]) C([B1.6])
void operatorPlacementNewWithConstructorWithinPlacementArgument() {
	new (new C()) C();
}

} // namespace operator_new
