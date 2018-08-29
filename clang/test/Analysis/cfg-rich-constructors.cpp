// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++11 -w %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX11,ELIDE,CXX11-ELIDE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++17 -w %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX17,ELIDE,CXX17-ELIDE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++11 -w -analyzer-config elide-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX11,NOELIDE,CXX11-NOELIDE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -std=c++17 -w -analyzer-config elide-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,CXX17,NOELIDE,CXX17-NOELIDE %s

class C {
public:
  C();
  C(C *);
  C(int, int);

  static C get();
  operator bool() const;
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
// CXX11-ELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.5])
// CXX11-NOELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4])
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], class C)
// CXX11-NEXT:     6: C c = C::get();
// CXX17-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4])
// CXX17-NEXT:     4: C c = C::get();
void simpleVariableInitializedByValue() {
  C c = C::get();
}

// FIXME: Find construction contexts for both branches in C++17.
// Note that once it gets detected, the test for the get() branch would not
// fail, because FileCheck allows partial matches.
// CHECK: void simpleVariableWithTernaryOperator(bool coin)
// CHECK:        [B1]
// CXX11-NEXT:     1: [B4.2] ? [B2.5] : [B3.6]
// CXX11-NEXT:     2: [B1.1]
// CXX11-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.4], class C)
// CXX11-NEXT:     4: C c = coin ? C::get() : C(0);
// CXX17-NEXT:     1: [B4.2] ? [B2.3] : [B3.4]
// CXX17-NEXT:     2: C c = coin ? C::get() : C(0);
// CHECK:        [B2]
// CHECK-NEXT:     1: C::get
// CHECK-NEXT:     2: [B2.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CXX11-ELIDE-NEXT:     3: [B2.2]() (CXXRecordTypedCall, [B2.4], [B2.5])
// CXX11-NOELIDE-NEXT:     3: [B2.2]() (CXXRecordTypedCall, [B2.4])
// CXX11-NEXT:     4: [B2.3]
// CXX11-ELIDE-NEXT:     5: [B2.4] (CXXConstructExpr, [B1.2], [B1.3], class C)
// CXX11-NOELIDE-NEXT:     5: [B2.4] (CXXConstructExpr, [B1.2], class C)
// CXX17-NEXT:     3: [B2.2]()
// CHECK:        [B3]
// CHECK-NEXT:     1: 0
// CHECK-NEXT:     2: [B3.1] (ImplicitCastExpr, NullToPointer, class C *)
// CXX11-ELIDE-NEXT:     3: [B3.2] (CXXConstructExpr, [B3.5], [B3.6], class C)
// CXX11-NOELIDE-NEXT:     3: [B3.2] (CXXConstructExpr, [B3.5], class C)
// CXX11-NEXT:     4: C([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CXX11-NEXT:     5: [B3.4]
// CXX11-ELIDE-NEXT:     6: [B3.5] (CXXConstructExpr, [B1.2], [B1.3], class C)
// CXX11-NOELIDE-NEXT:     6: [B3.5] (CXXConstructExpr, [B1.2], class C)
// CXX17-NEXT:     3: [B3.2] (CXXConstructExpr, class C)
// CXX17-NEXT:     4: C([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CHECK:        [B4]
// CHECK-NEXT:     1: coin
// CHECK-NEXT:     2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CHECK-NEXT:     T: [B4.2] ? ... : ...
void simpleVariableWithTernaryOperator(bool coin) {
  C c = coin ? C::get() : C(0);
}

// CHECK: void simpleVariableWithElidableCopy()
// CHECK:          1: 0
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NullToPointer, class C *)
// CXX11-ELIDE-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.5], [B1.6], class C)
// CXX11-NOELIDE-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.5], class C)
// CXX17-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     4: C([B1.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CXX11-NEXT:     5: [B1.4]
// CXX11-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.7], class C)
// CXX11-NEXT:     7: C c = C(0);
// CXX17-NEXT:     5: C c = C(0);
void simpleVariableWithElidableCopy() {
  C c = C(0);
}

// CHECK: void referenceVariableWithConstructor()
// CHECK:          1: 0
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NullToPointer, class C *)
// CHECK-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.4], const class C)
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: const C &c(0);
void referenceVariableWithConstructor() {
  const C &c(0);
}

// CHECK: void referenceVariableWithInitializer()
// CHECK:          1: C() (CXXConstructExpr, [B1.3], class C)
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class C)
// CHECK-NEXT:     3: [B1.2]
// CHECK-NEXT:     4: const C &c = C();
void referenceVariableWithInitializer() {
  const C &c = C();
}

// CHECK: void referenceVariableWithTernaryOperator(bool coin)
// CHECK:        [B1]
// CXX11-NEXT:     1: [B4.2] ? [B2.5] : [B3.6]
// CXX17-NEXT:     1: [B4.2] ? [B2.3] : [B3.4]
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class C)
// CHECK-NEXT:     3: [B1.2]
// CHECK-NEXT:     4: const C &c = coin ? C::get() : C(0);
// CHECK:        [B2]
// CHECK-NEXT:     1: C::get
// CHECK-NEXT:     2: [B2.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CXX11-ELIDE-NEXT:     3: [B2.2]() (CXXRecordTypedCall, [B2.4], [B2.5])
// CXX11-NOELIDE-NEXT:     3: [B2.2]() (CXXRecordTypedCall, [B2.4])
// CXX11-NEXT:     4: [B2.3]
// CXX11-NEXT:     5: [B2.4] (CXXConstructExpr, [B1.3], class C)
// CXX17-NEXT:     3: [B2.2]() (CXXRecordTypedCall, [B1.3])
// CHECK:        [B3]
// CHECK-NEXT:     1: 0
// CHECK-NEXT:     2: [B3.1] (ImplicitCastExpr, NullToPointer, class C *)
// CXX11-ELIDE-NEXT:     3: [B3.2] (CXXConstructExpr, [B3.5], [B3.6], class C)
// CXX11-NOELIDE-NEXT:     3: [B3.2] (CXXConstructExpr, [B3.5], class C)
// CXX11-NEXT:     4: C([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CXX11-NEXT:     5: [B3.4]
// CXX11-NEXT:     6: [B3.5] (CXXConstructExpr, [B1.3], class C)
// CXX17-NEXT:     3: [B3.2] (CXXConstructExpr, [B1.3], class C)
// CXX17-NEXT:     4: C([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
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

// FIXME: Why is CXXRecordTypedCall not present in C++17? Note that once it gets
// detected the test would not fail, because FileCheck allows partial matches.
// CHECK: D(double)
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CHECK-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4])
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: [B1.4] (CXXConstructExpr, C([B1.4]) (Base initializer), class C)
// CHECK-NEXT:     6: C([B1.5]) (Base initializer)
// CHECK-NEXT:     7: CFGNewAllocator(C *)
// CHECK-NEXT:     8: C::get
// CHECK-NEXT:     9: [B1.8] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CXX11-ELIDE-NEXT:    10: [B1.9]() (CXXRecordTypedCall, [B1.11], [B1.12])
// CXX11-NOELIDE-NEXT:    10: [B1.9]() (CXXRecordTypedCall, [B1.11])
// CXX11-NEXT:    11: [B1.10]
// CXX11-NEXT:    12: [B1.11] (CXXConstructExpr, [B1.13], class C)
// CXX11-NEXT:    13: new C([B1.12])
// CXX11-NEXT:    14: [B1.13] (CXXConstructExpr, c1([B1.13]) (Member initializer), class C)
// CXX11-NEXT:    15: c1([B1.14]) (Member initializer)
// CXX17-NEXT:    10: [B1.9]()
// CXX17-NEXT:    11: new C([B1.10])
// CXX17-NEXT:    12: [B1.11] (CXXConstructExpr, c1([B1.11]) (Member initializer), class C)
// CXX17-NEXT:    13: c1([B1.12]) (Member initializer)
  D(double): C(C::get()), c1(new C(C::get())) {}
};

// Let's see if initializers work well for fields with destructors.
class E {
public:
  static E get();
  ~E();
};

class F {
  E e;

public:
// FIXME: There should be no temporary destructor in C++17.
// CHECK: F()
// CHECK:          1: E::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class ctor_initializers::E (*)(
// CXX11-ELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.6], [B1.7])
// CXX11-NOELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.6])
// CXX11-NEXT:     4: [B1.3] (BindTemporary)
// CXX11-NEXT:     5: [B1.4] (ImplicitCastExpr, NoOp, const class ctor_initializers::E)
// CXX11-NEXT:     6: [B1.5]
// CXX11-NEXT:     7: [B1.6] (CXXConstructExpr, e([B1.6]) (Member initializer), class ctor_initializers
// CXX11-NEXT:     8: e([B1.7]) (Member initializer)
// CXX11-NEXT:     9: ~ctor_initializers::E() (Temporary object destructor)
// CXX17-NEXT:     3: [B1.2]() (CXXRecordTypedCall, e([B1.4]) (Member initializer), [B1.4])
// CXX17-NEXT:     4: [B1.3] (BindTemporary)
// CXX17-NEXT:     5: e([B1.4]) (Member initializer)
// CXX17-NEXT:     6: ~ctor_initializers::E() (Temporary object destructor)
  F(): e(E::get()) {}
};
} // end namespace ctor_initializers

namespace return_stmt_without_dtor {

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

// CHECK: C returnTemporary()
// CXX11-ELIDE:    1: C() (CXXConstructExpr, [B1.2], [B1.3], class C)
// CXX11-NOELIDE:  1: C() (CXXConstructExpr, [B1.2], class C)
// CXX11-NEXT:     2: [B1.1]
// CXX11-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.4], class C)
// CXX11-NEXT:     4: return [B1.3];
// CXX17:          1: C() (CXXConstructExpr, [B1.2], class C)
// CXX17-NEXT:     2: return [B1.1];
C returnTemporary() {
  return C();
}

// CHECK: C returnTemporaryWithArgument()
// CHECK:          1: nullptr
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NullToPointer, class C *)
// CXX11-ELIDE-NEXT: 3: [B1.2] (CXXConstructExpr, [B1.5], [B1.6], class C)
// CXX11-NOELIDE-NEXT: 3: [B1.2] (CXXConstructExpr, [B1.5], class C)
// CXX17-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     4: C([B1.3]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CXX11-NEXT:     5: [B1.4]
// CXX11-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.7], class C)
// CXX11-NEXT:     7: return [B1.6];
// CXX17-NEXT:     5: return [B1.4];

C returnTemporaryWithArgument() {
  return C(nullptr);
}

// CHECK: C returnTemporaryConstructedByFunction()
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CXX11-ELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.5])
// CXX11-NOELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4])
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], class C)
// CXX11-NEXT:     6: return [B1.5];
// CXX17-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4])
// CXX17-NEXT:     4: return [B1.3];
C returnTemporaryConstructedByFunction() {
  return C::get();
}

// CHECK: C returnChainOfCopies()
// CHECK:          1: C::get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class C (*)(void))
// CXX11-ELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.5])
// CXX11-NOELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4])
// CXX11-NEXT:     4: [B1.3]
// CXX11-ELIDE-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.7], [B1.8], class C)
// CXX11-NOELIDE-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.7], class C)
// CXX11-NEXT:     6: C([B1.5]) (CXXFunctionalCastExpr, ConstructorConversion, class C)
// CXX11-NEXT:     7: [B1.6]
// CXX11-NEXT:     8: [B1.7] (CXXConstructExpr, [B1.9], class C)
// CXX11-NEXT:     9: return [B1.8];
// CXX17-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.5])
// CXX17-NEXT:     4: C([B1.3]) (CXXFunctionalCastExpr, NoOp, class C)
// CXX17-NEXT:     5: return [B1.4];
C returnChainOfCopies() {
  return C(C::get());
}

} // end namespace return_stmt_without_dtor

namespace return_stmt_with_dtor {

class D {
public:
  D();
  ~D();
};

// FIXME: There should be no temporary destructor in C++17.
// CHECK:  return_stmt_with_dtor::D returnTemporary()
// CXX11-ELIDE:          1: return_stmt_with_dtor::D() (CXXConstructExpr, [B1.2], [B1.4], [B1.5], class return_stmt_with_dtor::D)
// CXX11-NOELIDE:          1: return_stmt_with_dtor::D() (CXXConstructExpr, [B1.2], [B1.4], class return_stmt_with_dtor::D)
// CXX11-NEXT:     2: [B1.1] (BindTemporary)
// CXX11-NEXT:     3: [B1.2] (ImplicitCastExpr, NoOp, const class return_stmt_with_dtor::D)
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.7], class return_stmt_with_dtor::D)
// CXX11-NEXT:     6: ~return_stmt_with_dtor::D() (Temporary object destructor)
// CXX11-NEXT:     7: return [B1.5];
// CXX17:          1: return_stmt_with_dtor::D() (CXXConstructExpr, [B1.4], [B1.2], class return_stmt_w
// CXX17-NEXT:     2: [B1.1] (BindTemporary)
// CXX17-NEXT:     3: ~return_stmt_with_dtor::D() (Temporary object destructor)
// CXX17-NEXT:     4: return [B1.2];
D returnTemporary() {
  return D();
}

// FIXME: There should be no temporary destructor in C++17.
// CHECK: void returnByValueIntoVariable()
// CHECK:          1: returnTemporary
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class return_stmt_with_dtor::D (*)(void))
// CXX11-ELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.6], [B1.7])
// CXX11-NOELIDE-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.4], [B1.6])
// CXX11-NEXT:     4: [B1.3] (BindTemporary)
// CXX11-NEXT:     5: [B1.4] (ImplicitCastExpr, NoOp, const class return_stmt_with_dtor::D)
// CXX11-NEXT:     6: [B1.5]
// CXX11-NEXT:     7: [B1.6] (CXXConstructExpr, [B1.8], class return_stmt_with_dtor::D)
// CXX11-NEXT:     8: return_stmt_with_dtor::D d = returnTemporary();
// CXX11-NEXT:     9: ~return_stmt_with_dtor::D() (Temporary object destructor)
// CXX11-NEXT:    10: [B1.8].~D() (Implicit destructor)
// CXX17-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.5], [B1.4])
// CXX17-NEXT:     4: [B1.3] (BindTemporary)
// CXX17-NEXT:     5: return_stmt_with_dtor::D d = returnTemporary();
// CXX17-NEXT:     6: ~return_stmt_with_dtor::D() (Temporary object destructor)
// CXX17-NEXT:     7: [B1.5].~D() (Implicit destructor)
void returnByValueIntoVariable() {
  D d = returnTemporary();
}

} // end namespace return_stmt_with_dtor

namespace temporary_object_expr_without_dtors {

// TODO: Should provide construction context for the constructor,
// even if there is no specific trigger statement here.
// CHECK: void simpleTemporary()
// CHECK           1: C() (CXXConstructExpr, class C)
void simpleTemporary() {
  C();
}

// CHECK: void temporaryInCondition()
// CHECK:          1: C() (CXXConstructExpr, [B2.2], class C)
// CHECK-NEXT:     2: [B2.1]
// CHECK-NEXT:     3: [B2.2] (ImplicitCastExpr, NoOp, const class C)
// CHECK-NEXT:     4: [B2.3].operator bool
// CHECK-NEXT:     5: [B2.3]
// CHECK-NEXT:     6: [B2.5] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK-NEXT:     T: if [B2.6]
void temporaryInCondition() {
  if (C());
}

// CHECK: void temporaryInConditionVariable()
// CXX11-ELIDE:    1: C() (CXXConstructExpr, [B2.2], [B2.3], class C)
// CXX11-NOELIDE:  1: C() (CXXConstructExpr, [B2.2], class C)
// CXX11-NEXT:     2: [B2.1]
// CXX11-NEXT:     3: [B2.2] (CXXConstructExpr, [B2.4], class C)
// CXX11-NEXT:     4: C c = C();
// CXX11-NEXT:     5: c
// CXX11-NEXT:     6: [B2.5] (ImplicitCastExpr, NoOp, const class C)
// CXX11-NEXT:     7: [B2.6].operator bool
// CXX11-NEXT:     8: [B2.6]
// CXX11-NEXT:     9: [B2.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX11-NEXT:     T: if [B2.9]
// CXX17:          1: C() (CXXConstructExpr, [B2.2], class C)
// CXX17-NEXT:     2: C c = C();
// CXX17-NEXT:     3: c
// CXX17-NEXT:     4: [B2.3] (ImplicitCastExpr, NoOp, const class C)
// CXX17-NEXT:     5: [B2.4].operator bool
// CXX17-NEXT:     6: [B2.4]
// CXX17-NEXT:     7: [B2.6] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX17-NEXT:     T: if [B2.7]
void temporaryInConditionVariable() {
  if (C c = C());
}


// CHECK: void temporaryInForLoopConditionVariable()
// CHECK:        [B2]
// CXX11-ELIDE-NEXT:     1: C() (CXXConstructExpr, [B2.2], [B2.3], class C)
// CXX11-NOELIDE-NEXT:     1: C() (CXXConstructExpr, [B2.2], class C)
// CXX11-NEXT:     2: [B2.1]
// CXX11-NEXT:     3: [B2.2] (CXXConstructExpr, [B2.4], class C)
// CXX11-NEXT:     4: C c2 = C();
// CXX11-NEXT:     5: c2
// CXX11-NEXT:     6: [B2.5] (ImplicitCastExpr, NoOp, const class C)
// CXX11-NEXT:     7: [B2.6].operator bool
// CXX11-NEXT:     8: [B2.6]
// CXX11-NEXT:     9: [B2.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX11-NEXT:     T: for (...; [B2.9]; )
// CXX17-NEXT:     1: C() (CXXConstructExpr, [B2.2], class C)
// CXX17-NEXT:     2: C c2 = C();
// CXX17-NEXT:     3: c2
// CXX17-NEXT:     4: [B2.3] (ImplicitCastExpr, NoOp, const class C)
// CXX17-NEXT:     5: [B2.4].operator bool
// CXX17-NEXT:     6: [B2.4]
// CXX17-NEXT:     7: [B2.6] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX17-NEXT:     T: for (...; [B2.7]; )
// CHECK:        [B3]
// CXX11-ELIDE-NEXT:     1: C() (CXXConstructExpr, [B3.2], [B3.3], class C)
// CXX11-NOELIDE-NEXT:     1: C() (CXXConstructExpr, [B3.2], class C)
// CXX11-NEXT:     2: [B3.1]
// CXX11-NEXT:     3: [B3.2] (CXXConstructExpr, [B3.4], class C)
// CXX11-NEXT:     4: C c1 = C();
// CXX17-NEXT:     1: C() (CXXConstructExpr, [B3.2], class C)
// CXX17-NEXT:     2: C c1 = C();
void temporaryInForLoopConditionVariable() {
  for (C c1 = C(); C c2 = C(); );
}


// CHECK: void temporaryInWhileLoopConditionVariable()
// CXX11-ELIDE:          1: C() (CXXConstructExpr, [B2.2], [B2.3], class C)
// CXX11-NOELIDE:          1: C() (CXXConstructExpr, [B2.2], class C)
// CXX11-NEXT:     2: [B2.1]
// CXX11-NEXT:     3: [B2.2] (CXXConstructExpr, [B2.4], class C)
// CXX11-NEXT:     4: C c = C();
// CXX11-NEXT:     5: c
// CXX11-NEXT:     6: [B2.5] (ImplicitCastExpr, NoOp, const class C)
// CXX11-NEXT:     7: [B2.6].operator bool
// CXX11-NEXT:     8: [B2.6]
// CXX11-NEXT:     9: [B2.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX11-NEXT:     T: while [B2.9]
// CXX17:          1: C() (CXXConstructExpr, [B2.2], class C)
// CXX17-NEXT:     2: C c = C();
// CXX17-NEXT:     3: c
// CXX17-NEXT:     4: [B2.3] (ImplicitCastExpr, NoOp, const class C)
// CXX17-NEXT:     5: [B2.4].operator bool
// CXX17-NEXT:     6: [B2.4]
// CXX17-NEXT:     7: [B2.6] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX17-NEXT:     T: while [B2.7]
void temporaryInWhileLoopConditionVariable() {
  while (C c = C());
}

} // end namespace temporary_object_expr_without_dtors

namespace temporary_object_expr_with_dtors {

class D {
public:
  D();
  D(int);
  ~D();

  static D get();

  operator bool() const;
};

// CHECK: void simpleTemporary()
// CHECK:          1: temporary_object_expr_with_dtors::D() (CXXConstructExpr, [B1.2], class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     2: [B1.1] (BindTemporary)
// CHECK-NEXT:     3: ~temporary_object_expr_with_dtors::D() (Temporary object destructor)
void simpleTemporary() {
  D();
}

// CHECK:  void temporaryInCondition()
// CHECK:          1: temporary_object_expr_with_dtors::D() (CXXConstructExpr, [B2.2], [B2.3], class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     2: [B2.1] (BindTemporary)
// CHECK-NEXT:     3: [B2.2]
// CHECK-NEXT:     4: [B2.3] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     5: [B2.4].operator bool
// CHECK-NEXT:     6: [B2.4]
// CHECK-NEXT:     7: [B2.6] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CHECK-NEXT:     8: ~temporary_object_expr_with_dtors::D() (Temporary object destructor)
// CHECK-NEXT:     T: if [B2.7]
void temporaryInCondition() {
  if (D());
}

// CHECK: void referenceVariableWithConstructor()
// CHECK:          1: 0
// CHECK-NEXT:     2: [B1.1] (CXXConstructExpr, [B1.4], const class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     3: [B1.2] (BindTemporary)
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: const temporary_object_expr_with_dtors::D &d(0);
// CHECK-NEXT:     6: [B1.5].~D() (Implicit destructor)
void referenceVariableWithConstructor() {
  const D &d(0);
}

// CHECK: void referenceVariableWithInitializer()
// CHECK:          1: temporary_object_expr_with_dtors::D() (CXXConstructExpr, [B1.4], class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     2: [B1.1] (BindTemporary)
// CHECK-NEXT:     3: [B1.2] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     4: [B1.3]
// CHECK-NEXT:     5: const temporary_object_expr_with_dtors::D &d = temporary_object_expr_with_dtors::D();
// CHECK-NEXT:     6: [B1.5].~D() (Implicit destructor)
void referenceVariableWithInitializer() {
  const D &d = D();
}

// CHECK: void referenceVariableWithTernaryOperator(bool coin)
// CXX11:        [B1]
// CXX11-NEXT:     1: [B4.4].~D() (Implicit destructor)
// CXX11:        [B2]
// CXX11-NEXT:     1: ~temporary_object_expr_with_dtors::D() (Temporary object destructor)
// CXX11:        [B3]
// CXX11-NEXT:     1: ~temporary_object_expr_with_dtors::D() (Temporary object destructor)
// CXX11:        [B4]
// CXX11-NEXT:     1: [B7.2] ? [B5.8] : [B6.8]
// CXX11-NEXT:     2: [B4.1] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     3: [B4.2]
// CXX11-NEXT:     4: const temporary_object_expr_with_dtors::D &d = coin ? D::get() : temporary_object_expr_with_dtors::D(0);
// CXX11-NEXT:     T: (Temp Dtor) [B6.3]
// CXX11:        [B5]
// CXX11-NEXT:     1: D::get
// CXX11-NEXT:     2: [B5.1] (ImplicitCastExpr, FunctionToPointerDecay, class temporary_object_expr_with_dtors::D (*)(void))
// CXX11-ELIDE-NEXT:     3: [B5.2]() (CXXRecordTypedCall, [B5.4], [B5.6], [B5.7])
// CXX11-NOELIDE-NEXT:     3: [B5.2]() (CXXRecordTypedCall, [B5.4], [B5.6])
// CXX11-NEXT:     4: [B5.3] (BindTemporary)
// CXX11-NEXT:     5: [B5.4] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     6: [B5.5]
// CXX11-NEXT:     7: [B5.6] (CXXConstructExpr, [B4.3], class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     8: [B5.7] (BindTemporary)
// CXX11:        [B6]
// CXX11-NEXT:     1: 0
// CXX11-ELIDE-NEXT:     2: [B6.1] (CXXConstructExpr, [B6.3], [B6.6], [B6.7], class temporary_object_expr_with_dtors::D)
// CXX11-NOELIDE-NEXT:     2: [B6.1] (CXXConstructExpr, [B6.3], [B6.6], class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     3: [B6.2] (BindTemporary)
// CXX11-NEXT:     4: temporary_object_expr_with_dtors::D([B6.3]) (CXXFunctionalCastExpr, ConstructorConversion, class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     5: [B6.4] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     6: [B6.5]
// CXX11-NEXT:     7: [B6.6] (CXXConstructExpr, [B4.3], class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     8: [B6.7] (BindTemporary)
// CXX11:        [B7]
// CXX11-NEXT:     1: coin
// CXX11-NEXT:     2: [B7.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CXX11-NEXT:     T: [B7.2] ? ... : ...
// CXX17:        [B1]
// CXX17-NEXT:     1: [B4.2] ? [B2.4] : [B3.4]
// CXX17-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CXX17-NEXT:     3: [B1.2]
// CXX17-NEXT:     4: const temporary_object_expr_with_dtors::D &d = coin ? D::get() : temporary_object_expr_with_dtors::D(0);
// CXX17-NEXT:     5: [B1.4].~D() (Implicit destructor)
// CXX17:        [B2]
// CXX17-NEXT:     1: D::get
// CXX17-NEXT:     2: [B2.1] (ImplicitCastExpr, FunctionToPointerDecay, class temporary_object_expr_with_dtors::D (*)(void))
// CXX17-NEXT:     3: [B2.2]() (CXXRecordTypedCall, [B1.3])
// CXX17-NEXT:     4: [B2.3] (BindTemporary)
// CXX17:        [B3]
// CXX17-NEXT:     1: 0
// CXX17-NEXT:     2: [B3.1] (CXXConstructExpr, [B1.3], class temporary_object_expr_with_dtors::D)
// CXX17-NEXT:     3: [B3.2] (BindTemporary)
// CXX17-NEXT:     4: temporary_object_expr_with_dtors::D([B3.3]) (CXXFunctionalCastExpr, ConstructorConversion, class temporary_object_expr_with_dtors::D)
// CXX17:        [B4]
// CXX17-NEXT:     1: coin
// CXX17-NEXT:     2: [B4.1] (ImplicitCastExpr, LValueToRValue, _Bool)
// CXX17-NEXT:     T: [B4.2] ? ... : ...
void referenceVariableWithTernaryOperator(bool coin) {
  const D &d = coin ? D::get() : D(0);
}

// CHECK: void referenceWithFunctionalCast()
// CHECK:          1: 1
// CHECK-NEXT:     2: [B1.1] (CXXConstructExpr, [B1.5], class temporary_object_expr_with_dtors::D)
// CHECK-NEXT:     3: [B1.2] (BindTemporary)
// CHECK-NEXT:     4: temporary_object_expr_with_dtors::D([B1.3]) (CXXFunctionalCastExpr, ConstructorCon
// CHECK-NEXT:     5: [B1.4]
// CHECK-NEXT:     6: temporary_object_expr_with_dtors::D &&d = temporary_object_expr_with_dtors::D(1);
// CHECK-NEXT:     7: [B1.6].~D() (Implicit destructor)
void referenceWithFunctionalCast() {
  D &&d = D(1);
}

// Test the condition constructor, we don't care about branch constructors here.
// CHECK: void constructorInTernaryCondition()
// CXX11:          1: 1
// CXX11-NEXT:     2: [B7.1] (CXXConstructExpr, [B7.3], [B7.5], class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     3: [B7.2] (BindTemporary)
// CXX11-NEXT:     4: temporary_object_expr_with_dtors::D([B7.3]) (CXXFunctionalCastExpr, ConstructorConversion, class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     5: [B7.4]
// CXX11-NEXT:     6: [B7.5] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CXX11-NEXT:     7: [B7.6].operator bool
// CXX11-NEXT:     8: [B7.6]
// CXX11-NEXT:     9: [B7.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX11-NEXT:     T: [B7.9] ? ... : ...
// CXX17:          1: 1
// CXX17-NEXT:     2: [B4.1] (CXXConstructExpr, [B4.3], [B4.5], class temporary_object_expr_with_dtors::D)
// CXX17-NEXT:     3: [B4.2] (BindTemporary)
// CXX17-NEXT:     4: temporary_object_expr_with_dtors::D([B4.3]) (CXXFunctionalCastExpr, ConstructorConversion, class temporary_object_expr_with_dtors::D)
// CXX17-NEXT:     5: [B4.4]
// CXX17-NEXT:     6: [B4.5] (ImplicitCastExpr, NoOp, const class temporary_object_expr_with_dtors::D)
// CXX17-NEXT:     7: [B4.6].operator bool
// CXX17-NEXT:     8: [B4.6]
// CXX17-NEXT:     9: [B4.8] (ImplicitCastExpr, UserDefinedConversion, _Bool)
// CXX17-NEXT:     T: [B4.9] ? ... : ...
void constructorInTernaryCondition() {
  const D &d = D(1) ? D(2) : D(3);
}

} // end namespace temporary_object_expr_with_dtors

namespace implicit_constructor_conversion {

class A {};
A get();

class B {
public:
  B(const A &);
  ~B() {}
};

// CHECK: void implicitConstructionConversionFromTemporary()
// CHECK:          1: implicit_constructor_conversion::A() (CXXConstructExpr, [B1.3], class implicit_constructor_conversion::A)
// CXX11-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::A)
// CXX11-NEXT:     3: [B1.2]
// CXX11-ELIDE-NEXT:     4: [B1.3] (CXXConstructExpr, [B1.6], [B1.8], [B1.9], class implicit_constructor_conversion::B)
// CXX11-NOELIDE-NEXT:     4: [B1.3] (CXXConstructExpr, [B1.6], [B1.8], class implicit_constructor_conversion::B)
// CXX11-NEXT:     5: [B1.4] (ImplicitCastExpr, ConstructorConversion, class implicit_constructor_conversion::B)
// CXX11-NEXT:     6: [B1.5] (BindTemporary)
// CXX11-NEXT:     7: [B1.6] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::B)
// CXX11-NEXT:     8: [B1.7]
// CXX11-NEXT:     9: [B1.8] (CXXConstructExpr, [B1.10], class implicit_constructor_conversion::B)
// CXX11-NEXT:    10: implicit_constructor_conversion::B b = implicit_constructor_conversion::A();
// CXX11-NEXT:    11: ~implicit_constructor_conversion::B() (Temporary object destructor)
// CXX11-NEXT:    12: [B1.10].~B() (Implicit destructor)
// CXX17-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::A)
// CXX17-NEXT:     3: [B1.2]
// CXX17-NEXT:     4: [B1.3] (CXXConstructExpr, [B1.6], class implicit_constructor_conversion::B)
// CXX17-NEXT:     5: [B1.4] (ImplicitCastExpr, ConstructorConversion, class implicit_constructor_conversion::B)
// CXX17-NEXT:     6: implicit_constructor_conversion::B b = implicit_constructor_conversion::A();
// CXX17-NEXT:     7: [B1.6].~B() (Implicit destructor)
void implicitConstructionConversionFromTemporary() {
  B b = A();
}

// CHECK: void implicitConstructionConversionFromFunctionValue()
// CHECK:          1: get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class implicit_constructor_conversion::A (*)(void))
// CHECK-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.5])
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::A)
// CHECK-NEXT:     5: [B1.4]
// CXX11-ELIDE-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.8], [B1.10], [B1.11], class implicit_constructor_conversion::B)
// CXX11-NOELIDE-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.8], [B1.10], class implicit_constructor_conversion::B)
// CXX11-NEXT:     7: [B1.6] (ImplicitCastExpr, ConstructorConversion, class implicit_constructor_conversion::B)
// CXX11-NEXT:     8: [B1.7] (BindTemporary)
// CXX11-NEXT:     9: [B1.8] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::B)
// CXX11-NEXT:    10: [B1.9]
// CXX11-NEXT:    11: [B1.10] (CXXConstructExpr, [B1.12], class implicit_constructor_conversion::B)
// CXX11-NEXT:    12: implicit_constructor_conversion::B b = get();
// CXX11-NEXT:    13: ~implicit_constructor_conversion::B() (Temporary object destructor)
// CXX11-NEXT:    14: [B1.12].~B() (Implicit destructor)
// CXX17-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.8], class implicit_constructor_conversion::B)
// CXX17-NEXT:     7: [B1.6] (ImplicitCastExpr, ConstructorConversion, class implicit_constructor_conversion::B)
// CXX17-NEXT:     8: implicit_constructor_conversion::B b = get();
// CXX17-NEXT:     9: [B1.8].~B() (Implicit destructor)
void implicitConstructionConversionFromFunctionValue() {
  B b = get();
}

// CHECK: void implicitConstructionConversionFromTemporaryWithLifetimeExtension()
// CHECK:          1: implicit_constructor_conversion::A() (CXXConstructExpr, [B1.3], class implicit_constructor_conversion::A)
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::A)
// CHECK-NEXT:     3: [B1.2]
// CHECK-NEXT:     4: [B1.3] (CXXConstructExpr, [B1.7], class implicit_constructor_conversion::B)
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, ConstructorConversion, class implicit_constructor_conversion::B)
// CHECK-NEXT:     6: [B1.5] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::B)
// CHECK-NEXT:     7: [B1.6]
// CHECK-NEXT:     8: const implicit_constructor_conversion::B &b = implicit_constructor_conversion::A();
// CHECK-NEXT:     9: [B1.8].~B() (Implicit destructor)
void implicitConstructionConversionFromTemporaryWithLifetimeExtension() {
  const B &b = A();
}

// CHECK: void implicitConstructionConversionFromFunctionValueWithLifetimeExtension()
// CHECK:          1: get
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class implicit_constructor_conver
// CHECK-NEXT:     3: [B1.2]() (CXXRecordTypedCall, [B1.5])
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::A)
// CHECK-NEXT:     5: [B1.4]
// CHECK-NEXT:     6: [B1.5] (CXXConstructExpr, [B1.9], class implicit_constructor_conversion::B)
// CHECK-NEXT:     7: [B1.6] (ImplicitCastExpr, ConstructorConversion, class implicit_constructor_convers
// CHECK-NEXT:     8: [B1.7] (ImplicitCastExpr, NoOp, const class implicit_constructor_conversion::B)
// CHECK-NEXT:     9: [B1.8]
// CHECK-NEXT:    10: const implicit_constructor_conversion::B &b = get();
// CHECK-NEXT:    11: [B1.10].~B() (Implicit destructor)
void implicitConstructionConversionFromFunctionValueWithLifetimeExtension() {
  const B &b = get(); // no-crash
}

} // end namespace implicit_constructor_conversion

namespace argument_constructors {
class D {
public:
  D();
  ~D();
};

class E {
public:
  E(D d);
  E(D d1, D d2);
};

void useC(C c);
void useCByReference(const C &c);
void useD(D d);
void useDByReference(const D &d);
void useCAndD(C c, D d);

// CHECK: void passArgument()
// CHECK:          1: useC
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(class C))
// CXX11-ELIDE-NEXT:     3: C() (CXXConstructExpr, [B1.4], [B1.5], class C)
// CXX11-NOELIDE-NEXT:     3: C() (CXXConstructExpr, [B1.4], class C)
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6]+0, class C)
// CXX11-NEXT:     6: [B1.2]([B1.5])
// CXX17-NEXT:     3: C() (CXXConstructExpr, [B1.4]+0, class C)
// CXX17-NEXT:     4: [B1.2]([B1.3])
void passArgument() {
  useC(C());
}

// CHECK: void passTwoArguments()
// CHECK:        [B1]
// CHECK-NEXT:     1: useCAndD
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(class C, class argument_constructors::D))
// CXX11-ELIDE-NEXT:     3: C() (CXXConstructExpr, [B1.4], [B1.5], class C)
// CXX11-NOELIDE-NEXT:     3: C() (CXXConstructExpr, [B1.4], class C)
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.12]+0, class C)
// CXX11-ELIDE-NEXT:     6: argument_constructors::D() (CXXConstructExpr, [B1.7], [B1.9], [B1.10], class argument_constructors::D)
// CXX11-NOELIDE-NEXT:     6: argument_constructors::D() (CXXConstructExpr, [B1.7], [B1.9], class argument_constructors::D)
// CXX11-NEXT:     7: [B1.6] (BindTemporary)
// CXX11-NEXT:     8: [B1.7] (ImplicitCastExpr, NoOp, const class argument_constructors::D)
// CXX11-NEXT:     9: [B1.8]
// CXX11-NEXT:    10: [B1.9] (CXXConstructExpr, [B1.11], [B1.12]+1, class argument_constructors::D)
// CXX11-NEXT:    11: [B1.10] (BindTemporary)
// CXX11-NEXT:    12: [B1.2]([B1.5], [B1.11])
// CXX11-NEXT:    13: ~argument_constructors::D() (Temporary object destructor)
// CXX11-NEXT:    14: ~argument_constructors::D() (Temporary object destructor)
// CXX17-NEXT:     3: C() (CXXConstructExpr, [B1.6]+0, class C)
// CXX17-NEXT:     4: argument_constructors::D() (CXXConstructExpr, [B1.5], [B1.6]+1, class argument_co
// CXX17-NEXT:     5: [B1.4] (BindTemporary)
// CXX17-NEXT:     6: [B1.2]([B1.3], [B1.5])
// CXX17-NEXT:     7: ~argument_constructors::D() (Temporary object destructor)
void passTwoArguments() {
  useCAndD(C(), D());
}

// CHECK: void passArgumentByReference()
// CHECK:          1: useCByReference
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(const class C &))
// CHECK-NEXT:     3: C() (CXXConstructExpr, [B1.5], class C)
// CHECK-NEXT:     4: [B1.3] (ImplicitCastExpr, NoOp, const class C)
// CHECK-NEXT:     5: [B1.4]
// CHECK-NEXT:     6: [B1.2]([B1.5])
void passArgumentByReference() {
  useCByReference(C());
}

// CHECK: void passArgumentWithDestructor()
// CHECK:          1: useD
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(class argument_constructors::D))
// CXX11-ELIDE-NEXT:     3: argument_constructors::D() (CXXConstructExpr, [B1.4], [B1.6], [B1.7], class argument_constructors::D)
// CXX11-NOELIDE-NEXT:     3: argument_constructors::D() (CXXConstructExpr, [B1.4], [B1.6], class argument_constructors::D)
// CXX11-NEXT:     4: [B1.3] (BindTemporary)
// CXX11-NEXT:     5: [B1.4] (ImplicitCastExpr, NoOp, const class argument_constructors::D)
// CXX11-NEXT:     6: [B1.5]
// CXX11-NEXT:     7: [B1.6] (CXXConstructExpr, [B1.8], [B1.9]+0, class argument_constructors::D)
// CXX11-NEXT:     8: [B1.7] (BindTemporary)
// CXX11-NEXT:     9: [B1.2]([B1.8])
// CXX11-NEXT:    10: ~argument_constructors::D() (Temporary object destructor)
// CXX11-NEXT:    11: ~argument_constructors::D() (Temporary object destructor)
// CXX17-NEXT:     3: argument_constructors::D() (CXXConstructExpr, [B1.4], [B1.5]+0, class argument_constructors::D)
// CXX17-NEXT:     4: [B1.3] (BindTemporary)
// CXX17-NEXT:     5: [B1.2]([B1.4])
// CXX17-NEXT:     6: ~argument_constructors::D() (Temporary object destructor)
void passArgumentWithDestructor() {
  useD(D());
}

// CHECK: void passArgumentWithDestructorByReference()
// CHECK:          1: useDByReference
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, void (*)(const class argumen
// CHECK-NEXT:     3: argument_constructors::D() (CXXConstructExpr, [B1.4], [B1.6], class argument_c
// CHECK-NEXT:     4: [B1.3] (BindTemporary)
// CHECK-NEXT:     5: [B1.4] (ImplicitCastExpr, NoOp, const class argument_constructors::D)
// CHECK-NEXT:     6: [B1.5]
// CHECK-NEXT:     7: [B1.2]([B1.6])
// CHECK-NEXT:     8: ~argument_constructors::D() (Temporary object destructor)
void passArgumentWithDestructorByReference() {
  useDByReference(D());
}

// CHECK: void passArgumentIntoAnotherConstructor()
// CXX11-ELIDE:          1: argument_constructors::D() (CXXConstructExpr, [B1.2], [B1.4], [B1.5], class argument_constructors::D)
// CXX11-NOELIDE:          1: argument_constructors::D() (CXXConstructExpr, [B1.2], [B1.4], class argument_constructors::D)
// CXX11-NEXT:     2: [B1.1] (BindTemporary)
// CXX11-NEXT:     3: [B1.2] (ImplicitCastExpr, NoOp, const class argument_constructors::D)
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], [B1.7]+0, class argument_constructors::D)
// CXX11-NEXT:     6: [B1.5] (BindTemporary)
// CXX11-ELIDE-NEXT:     7: [B1.6] (CXXConstructExpr, [B1.9], [B1.10], class argument_constructors::E)
// CXX11-NOELIDE-NEXT:     7: [B1.6] (CXXConstructExpr, [B1.9], class argument_constructors::E)
// CXX11-NEXT:     8: argument_constructors::E([B1.7]) (CXXFunctionalCastExpr, ConstructorConversion, class argument_constructors::E)
// CXX11-NEXT:     9: [B1.8]
// CXX11-NEXT:    10: [B1.9] (CXXConstructExpr, [B1.11], class argument_constructors::E)
// CXX11-NEXT:    11: argument_constructors::E e = argument_constructors::E(argument_constructors::D());
// CXX11-NEXT:    12: ~argument_constructors::D() (Temporary object destructor)
// CXX11-NEXT:    13: ~argument_constructors::D() (Temporary object destructor)
// CXX17:          1: argument_constructors::D() (CXXConstructExpr, [B1.2], [B1.3]+0, class argument_constructors::D)
// CXX17-NEXT:     2: [B1.1] (BindTemporary)
// CXX17-NEXT:     3: [B1.2] (CXXConstructExpr, [B1.5], class argument_constructors::E)
// CXX17-NEXT:     4: argument_constructors::E([B1.3]) (CXXFunctionalCastExpr, ConstructorConversion, class argument_constructors::E)
// CXX17-NEXT:     5: argument_constructors::E e = argument_constructors::E(argument_constructors::D());
// CXX17-NEXT:     6: ~argument_constructors::D() (Temporary object destructor)
void passArgumentIntoAnotherConstructor() {
  E e = E(D());
}


// CHECK: void passTwoArgumentsIntoAnotherConstructor()
// CXX11-ELIDE:          1: argument_constructors::D() (CXXConstructExpr, [B1.2], [B1.4], [B1.5], class argument_constructors::D)
// CXX11-NOELIDE:          1: argument_constructors::D() (CXXConstructExpr, [B1.2], [B1.4], class argument_constructors::D)
// CXX11-NEXT:     2: [B1.1] (BindTemporary)
// CXX11-NEXT:     3: [B1.2] (ImplicitCastExpr, NoOp, const class argument_constructors::D)
// CXX11-NEXT:     4: [B1.3]
// CXX11-NEXT:     5: [B1.4] (CXXConstructExpr, [B1.6], [B1.13]+0, class argument_constructors::D)
// CXX11-NEXT:     6: [B1.5] (BindTemporary)
// CXX11-ELIDE-NEXT:     7: argument_constructors::D() (CXXConstructExpr, [B1.8], [B1.10], [B1.11], class argument_constructors::D)
// CXX11-NOELIDE-NEXT:     7: argument_constructors::D() (CXXConstructExpr, [B1.8], [B1.10], class argument_constructors::D)
// CXX11-NEXT:     8: [B1.7] (BindTemporary)
// CXX11-NEXT:     9: [B1.8] (ImplicitCastExpr, NoOp, const class argument_constructors::D)
// CXX11-NEXT:    10: [B1.9]
// CXX11-NEXT:    11: [B1.10] (CXXConstructExpr, [B1.12], [B1.13]+1, class argument_constructors::D)
// CXX11-NEXT:    12: [B1.11] (BindTemporary)
// CXX11-NEXT:    13: argument_constructors::E([B1.6], [B1.12]) (CXXConstructExpr, class argument_constructors::E)
// CXX11-NEXT:    14: ~argument_constructors::D() (Temporary object destructor)
// CXX11-NEXT:    15: ~argument_constructors::D() (Temporary object destructor)
// CXX11-NEXT:    16: ~argument_constructors::D() (Temporary object destructor)
// CXX11-NEXT:    17: ~argument_constructors::D() (Temporary object destructor)
// CXX17:          1: argument_constructors::D() (CXXConstructExpr, [B1.2], [B1.5]+0, class argument_constructors::D)
// CXX17-NEXT:     2: [B1.1] (BindTemporary)
// CXX17-NEXT:     3: argument_constructors::D() (CXXConstructExpr, [B1.4], [B1.5]+1, class argument_constructors::D)
// CXX17-NEXT:     4: [B1.3] (BindTemporary)
// CXX17-NEXT:     5: argument_constructors::E([B1.2], [B1.4]) (CXXConstructExpr, class argument_constructors::E)
// CXX17-NEXT:     6: ~argument_constructors::D() (Temporary object destructor)
// CXX17-NEXT:     7: ~argument_constructors::D() (Temporary object destructor)
void passTwoArgumentsIntoAnotherConstructor() {
  E(D(), D());
}
} // end namespace argument_constructors

namespace copy_elision_with_extra_arguments {
class C {
public:
  C();
  C(const C &c, int x = 0);
};

// CHECK: void testCopyElisionWhenCopyConstructorHasExtraArguments()
// CHECK:        [B1]
// CXX11-ELIDE-NEXT:     1: copy_elision_with_extra_arguments::C() (CXXConstructExpr, [B1.3], [B1.5], class copy_elision_with_extra_arguments::C)
// CXX11-NOELIDE-NEXT:     1: copy_elision_with_extra_arguments::C() (CXXConstructExpr, [B1.3], class copy_elision_with_extra_arguments::C)
// CXX11-NEXT:     2: [B1.1] (ImplicitCastExpr, NoOp, const class copy_elision_with_extra_arguments::C)
// CXX11-NEXT:     3: [B1.2]
// CXX11-NEXT:     4:
// CXX11-NEXT:     5: [B1.3] (CXXConstructExpr, [B1.6], class copy_elision_with_extra_arguments::C)
// CXX11-NEXT:     6: copy_elision_with_extra_arguments::C c = copy_elision_with_extra_arguments::C();
// CXX17-NEXT:     1: copy_elision_with_extra_arguments::C() (CXXConstructExpr, [B1.2], class copy_elision_with_extra_arguments::C)
// CXX17-NEXT:     2: copy_elision_with_extra_arguments::C c = copy_elision_with_extra_arguments::C();
void testCopyElisionWhenCopyConstructorHasExtraArguments() {
  C c = C();
}
} // namespace copy_elision_with_extra_arguments


namespace operators {
class C {
public:
  C(int);
  C &operator+(C Other);
};

// FIXME: Find construction context for the this-argument of the operator.
// CHECK: void testOperators()
// CHECK:        [B1]
// CHECK-NEXT:     1: operator+
// CHECK-NEXT:     2: [B1.1] (ImplicitCastExpr, FunctionToPointerDecay, class operators::C &(*)(class o
// CHECK-NEXT:     3: 1
// CHECK-NEXT:     4: [B1.3] (CXXConstructExpr, [B1.6], class operators::C)
// CHECK-NEXT:     5: operators::C([B1.4]) (CXXFunctionalCastExpr, ConstructorConversion, class operato
// CHECK-NEXT:     6: [B1.5]
// CHECK-NEXT:     7: 2
// CXX11-ELIDE-NEXT:     8: [B1.7] (CXXConstructExpr, [B1.10], [B1.11], class operators::C)
// CXX11-NOELIDE-NEXT:     8: [B1.7] (CXXConstructExpr, [B1.10], class operators::C)
// CXX11-NEXT:     9: operators::C([B1.8]) (CXXFunctionalCastExpr, ConstructorConversion, class operato
// CXX11-NEXT:    10: [B1.9]
// CXX11-NEXT:    11: [B1.10] (CXXConstructExpr, [B1.12]+1, class operators::C)
// CXX11-NEXT:    12: [B1.6] + [B1.11] (OperatorCall)
// CXX17-NEXT:     8: [B1.7] (CXXConstructExpr, [B1.10]+1, class operators::C)
// CXX17-NEXT:     9: operators::C([B1.8]) (CXXFunctionalCastExpr, ConstructorConversion, class operato
// CXX17-NEXT:    10: [B1.6] + [B1.9] (OperatorCall)
void testOperators() {
  C(1) + C(2);
}
} // namespace operators

namespace variadic_function_arguments {
class C {
 public:
  C(int);
};

int variadic(...);

// This code is quite exotic, so let's not test the CFG for it,
// but only make sure we don't crash.
void testCrashOnVariadicArgument() {
  C c(variadic(0 ? c : 0)); // no-crash
}
} // namespace variadic_function_arguments
