// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -analyzer-config cfg-temporary-dtors=true -std=c++11 %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

class C {
public:
  C();
  C(C *);
  C(int, int);

  static C get();
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

namespace decl_stmt {

// CHECK: void simpleVariable()
// CHECK:          1:  (CXXConstructExpr, [B1.2], class C)
// CHECK-NEXT:     2: C c;
void simpleVariable() {
  C c;
}

// CHECK: void simpleVariableWithBraces()
// CHECK:          1: {} (CXXConstructExpr, [B1.2], class C)
// CHECK-NEXT:     2: C c{};
void simpleVariableWithBraces() {
  C c{};
}

// CHECK: void simpleVariableWithConstructorArgument()
// CHECK:          1: 0
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NullToPointer, class C *)
// CHECK-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.4], class C)
// CHECK-NEXT:     4: C c(0);
void simpleVariableWithConstructorArgument() {
  C c(0);
}

// CHECK: void simpleVariableWithOperatorNewInConstructorArgument()
// CHECK:          1: CFGNewAllocator(C *)
// CHECK-NEXT:     2:  (CXXConstructExpr, [B1.3], class C)
// CHECK-NEXT:     3: new C([B1.2])
// CHECK-NEXT:     4: [B1.3] (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     5: C c(new C());
void simpleVariableWithOperatorNewInConstructorArgument() {
  C c(new C());
}

// CHECK: void simpleVariableWithOperatorNewInBraces()
// CHECK:          1: CFGNewAllocator(C *)
// CHECK-NEXT:     2:  (CXXConstructExpr, [B1.3], class C)
// CHECK-NEXT:     3: new C([B1.2])
// CHECK-NEXT:     4: {[B1.3]} (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     5: C c{new C()};
void simpleVariableWithOperatorNewInBraces() {
  C c{new C()};
}

// CHECK: void simpleVariableInitializedByValue()
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], class C)
// CHECK-NEXT:     6: C c = C::get();
void simpleVariableInitializedByValue() {
  C c = C::get();
}

// TODO: Should find construction target for the three temporaries as well.
// CHECK: void simpleVariableWithTernaryOperator(bool coin)
// CHECK:        [B1]
// CHECK-NEXT:     1: [B4.2] ? [B2.5] : [B3.6]
// CHECK-NEXT:     2: [B1.1]
// CHECK-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.4], class C)
// CHECK-NEXT:     4: C c = coin ? C::get() : C(0);
// CHECK:        [B2]
// CHECK-NEXT:     1: C::get
// CHECK-NEXT:     2: [B2.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B2.2]()
// CHECK-NEXT:     4: [B2.3]
// CHECK-NEXT:     5: [B2.4] (CXXConstructExpr, class C)
// CHECK:        [B3]
// CHECK-NEXT:     1: 0
// CHECK-NEXT:     2: [B3.1] (ImplicitCastExpr, NullToPointer, class C *)
// CHECK-NEXT:     3: [B3.2] (CXXConstructExpr, class C)
// CHECK-NEXT:     4: C([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CHECK-NEXT:     5: [B3.4]
// CHECK-NEXT:     6: [B3.5] (CXXConstructExpr, class C)
// CHECK:        [B4]
// CHECK-NEXT:     1: coin
// CHECK-NEXT:     2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:     T: [B4.2] ? ... : ...
void simpleVariableWithTernaryOperator(bool coin) {
  C c = coin ? C::get() : C(0);
}

// TODO: Should find construction target here.
// CHECK: void referenceVariableWithConstructor()
// CHECK:          1: 0
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NullToPointer, class C *)
// CHECK-NEXT:     3: [B1.2] (CXXConstructExpr, const class C)
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: const C &c(0);
void referenceVariableWithConstructor() {
  const C &c(0);
}

// TODO: Should find construction target here.
// CHECK: void referenceVariableWithInitializer()
// CHECK:          1: C() (CXXConstructExpr, class C)
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class C)
// CHECK-NEXT:     3: [B1.2]
// CHECK-NEXT:     4: const C &c = C();
void referenceVariableWithInitializer() {
  const C &c = C();
}

// TODO: Should find construction targets here.
// CHECK: void referenceVariableWithTernaryOperator(bool coin)
// CHECK:        [B1]
// CHECK-NEXT:     1: [B4.2] ? [B2.5] : [B3.6]
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class C)
// CHECK-NEXT:     3: [B1.2]
// CHECK-NEXT:     4: const C &c = coin ? C::get() : C(0);
// CHECK:        [B2]
// CHECK-NEXT:     1: C::get
// CHECK-NEXT:     2: [B2.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B2.2]()
// CHECK-NEXT:     4: [B2.3]
// CHECK-NEXT:     5: [B2.4] (CXXConstructExpr, class C)
// CHECK:        [B3]
// CHECK-NEXT:     1: 0
// CHECK-NEXT:     2: [B3.1] (ImplicitCastExpr, NullToPointer, class C *)
// CHECK-NEXT:     3: [B3.2] (CXXConstructExpr, class C)
// CHECK-NEXT:     4: C([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CHECK-NEXT:     5: [B3.4]
// CHECK-NEXT:     6: [B3.5] (CXXConstructExpr, class C)
// CHECK:        [B4]
// CHECK-NEXT:     1: coin
// CHECK-NEXT:     2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:     T: [B4.2] ? ... : ...
void referenceVariableWithTernaryOperator(bool coin) {
  const C &c = coin ? C::get() : C(0);
}

} // end namespace decl_stmt

namespace ctor_initializers {

class D: public C {
  C c1;

public:

// CHECK: D()
// CHECK:          1:  (CXXConstructExpr, C() (Base initializer), class C)
// CHECK-NEXT:     2: C([B1.1]) (Base initializer)
// CHECK-NEXT:     3: CFGNewAllocator(C *)
// CHECK-NEXT:     4:  (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     5: new C([B1.4])
// CHECK-NEXT:     6: [B1.5] (CXXConstructExpr, c1([B1.5]) (Member initializer), class C)
// CHECK-NEXT:     7: c1([B1.6]) (Member initializer)
  D(): C(), c1(new C()) {}

// CHECK: D(int)
// CHECK:          1:  (CXXConstructExpr, D() (Delegating initializer), class ctor_initializers::D)
// CHECK-NEXT:     2: D([B1.1]) (Delegating initializer)
  D(int): D() {}

// CHECK: D(double)
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, C([B1.4]) (Base initializer), class C)
// CHECK-NEXT:     6: C([B1.5]) (Base initializer)
// CHECK-NEXT:     7: CFGNewAllocator(C *)
// CHECK-NEXT:     8: C::get
// CHECK-NEXT:     9: [B1.8] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:    10: [B1.9]()
// CHECK-NEXT:    11: [B1.10]
// CHECK-NEXT:    12: [B1.11] (CXXConstructExpr, [B1.13], class C)
// CHECK-NEXT:    13: new C([B1.12])
// CHECK-NEXT:    14: [B1.13] (CXXConstructExpr, c1([B1.13]) (Member initializer), class C)
// CHECK-NEXT:    15: c1([B1.14]) (Member initializer)
  D(double): C(C::get()), c1(new C(C::get())) {}
};

} // end namespace ctor_initializers

namespace return_stmt {

// CHECK: C returnVariable()
// CHECK:          1:  (CXXConstructExpr, [B1.2], class C)
// CHECK-NEXT:     2: C c;
// CHECK-NEXT:     3: c
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, NoOp, class C)
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], class C)
// CHECK-NEXT:     6: return [B1.5];
C returnVariable() {
  C c;
  return c;
}

// CHECK: C returnEmptyBraces()
// CHECK:          1: {} (CXXConstructExpr, [B1.2], class C)
// CHECK-NEXT:     2: return [B1.1];
C returnEmptyBraces() {
  return {};
}

// CHECK: C returnBracesWithOperatorNew()
// CHECK:          1: CFGNewAllocator(C *)
// CHECK-NEXT:     2:  (CXXConstructExpr, [B1.3], class C)
// CHECK-NEXT:     3: new C([B1.2])
// CHECK-NEXT:     4: {[B1.3]} (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     5: return [B1.4];
C returnBracesWithOperatorNew() {
  return {new C()};
}

// CHECK: C returnBracesWithMultipleItems()
// CHECK:          1: 123
// CHECK-NEXT:     2: 456
// CHECK-NEXT:     3: {[B1.1], [B1.2]} (CXXConstructExpr, [B1.4], class C)
// CHECK-NEXT:     4: return [B1.3];
C returnBracesWithMultipleItems() {
  return {123, 456};
}

// TODO: Should find construction targets for the first constructor as well.
// CHECK: C returnTemporary()
// CHECK:          1: C() (CXXConstructExpr, class C)
// CHECK-NEXT:     2: [B1.1]
// CHECK-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.4], class C)
// CHECK-NEXT:     4: return [B1.3];
C returnTemporary() {
  return C();
}

// TODO: Should find construction targets for the first constructor as well.
// CHECK: C returnTemporaryWithArgument()
// CHECK:          1: nullptr
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NullToPointer, class C *)
// CHECK-NEXT:     3: [B1.2] (CXXConstructExpr, class C)
// CHECK-NEXT:     4: C([B1.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CHECK-NEXT:     5: [B1.4]
// CHECK-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.7], class C)
// CHECK-NEXT:     7: return [B1.6];
C returnTemporaryWithArgument() {
  return C(nullptr);
}

// CHECK: C returnTemporaryConstructedByFunction()
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], class C)
// CHECK-NEXT:     6: return [B1.5];
C returnTemporaryConstructedByFunction() {
  return C::get();
}

// TODO: Should find construction targets for the first constructor as well.
// CHECK: C returnChainOfCopies()
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B1.2]()
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, class C)
// CHECK-NEXT:     6: C([B1.5]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CHECK-NEXT:     7: [B1.6]
// CHECK-NEXT:     8: [B1.7] (CXXConstructExpr, [B1.9], class C)
// CHECK-NEXT:     9: return [B1.8];
C returnChainOfCopies() {
  return C(C::get());
}

} // end namespace return_stmt
