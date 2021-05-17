// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -fheinous-gnu-extensions -std=c++11 -analyzer-config cfg-rich-constructors=false %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,WARNINGS %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -triple x86_64-apple-darwin12 -fheinous-gnu-extensions -std=c++11 -analyzer-config cfg-rich-constructors=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t -check-prefixes=CHECK,ANALYZER %s

// This file tests how we construct two different flavors of the Clang CFG -
// the CFG used by the Sema analysis-based warnings and the CFG used by the
// static analyzer. The difference in the behavior is checked via FileCheck
// prefixes (WARNINGS and ANALYZER respectively). When introducing new analyzer
// flags, no new run lines should be added - just these flags would go to the
// respective line depending on where is it turned on and where is it turned
// off. Feel free to add tests that test only one of the CFG flavors if you're
// not sure how the other flavor is supposed to work in your case.

// CHECK-LABEL: void checkDeclStmts()
// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: int i;
// CHECK-NEXT:   2: int j;
// CHECK-NEXT:   3: 1
// CHECK-NEXT:   4: int k = 1;
// CHECK-NEXT:   5: int l;
// CHECK-NEXT:   6: 2
// CHECK-NEXT:   7: int m = 2;
// WARNINGS-NEXT: (CXXConstructExpr, struct standalone)
// ANALYZER-NEXT: (CXXConstructExpr, [B1.9], struct standalone)
// CHECK-NEXT:   9: struct standalone myStandalone;
// WARNINGS-NEXT: (CXXConstructExpr, struct (unnamed struct at {{.*}}))
// ANALYZER-NEXT: (CXXConstructExpr, [B1.11], struct (unnamed struct at {{.*}}))
// CHECK-NEXT:  11: struct (unnamed struct at {{.*}}) myAnon;
// WARNINGS-NEXT: (CXXConstructExpr, struct named)
// ANALYZER-NEXT: (CXXConstructExpr, [B1.13], struct named)
// CHECK-NEXT:  13: struct named myNamed;
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
void checkDeclStmts() {
  int i, j;
  int k = 1, l, m = 2;

  struct standalone { int x, y; };
  struct standalone myStandalone;

  struct { int x, y; } myAnon;

  struct named { int x, y; } myNamed;

  static_assert(1, "abc");
}

// CHECK-LABEL: void F(EmptyE e)
// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: e
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, enum EmptyE)
// CHECK-NEXT:   3: [B1.2] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:   T: switch [B1.3]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
enum EmptyE {};
void F(EmptyE e) {
  switch (e) {}
}

// CHECK-LABEL: void testBuiltinSize()
// CHECK: ENTRY
// CHECK-NEXT: Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: __builtin_object_size
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, BuiltinFnToFnPtr, unsigned long (*)(const void *, int) noexcept)
// CHECK-NEXT:   3: [B1.2](dummy(), 0)
// CHECK-NEXT:   4: (void)[B1.3] (CStyleCastExpr, ToVoid, void)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void testBuiltinSize() {
  extern int *dummy();
  (void)__builtin_object_size(dummy(), 0);
}

class A {
public:
  A() {}
  ~A() {}
};

// CHECK-LABEL: void test_deletedtor()
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1:  CFGNewAllocator(A *)
// WARNINGS-NEXT:   2:  (CXXConstructExpr, class A)
// ANALYZER-NEXT:   2:  (CXXConstructExpr, [B1.3], class A)
// CHECK-NEXT:   3: new A([B1.2])
// CHECK-NEXT:   4: A *a = new A();
// CHECK-NEXT:   5: a
// CHECK-NEXT:   6: [B1.5] (ImplicitCastExpr, LValueToRValue, class A *)
// CHECK-NEXT:   7: [B1.6]->~A() (Implicit destructor)
// CHECK-NEXT:   8: delete [B1.6]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_deletedtor() {
  A *a = new A();
  delete a;
}

// CHECK-LABEL: void test_deleteArraydtor()
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: 5
// CHECK-NEXT:   2: CFGNewAllocator(A *)
// WARNINGS-NEXT:   3:  (CXXConstructExpr, class A [5])
// ANALYZER-NEXT:   3:  (CXXConstructExpr, [B1.4], class A [5])
// CHECK-NEXT:   4: new A {{\[\[}}B1.1]]
// CHECK-NEXT:   5: A *a = new A [5];
// CHECK-NEXT:   6: a
// CHECK-NEXT:   7: [B1.6] (ImplicitCastExpr, LValueToRValue, class A *)
// CHECK-NEXT:   8: [B1.7]->~A() (Implicit destructor)
// CHECK-NEXT:   9: delete [] [B1.7]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void test_deleteArraydtor() {
  A *a = new A[5];
  delete[] a;
}


namespace NoReturnSingleSuccessor {
  struct A {
    A();
    ~A();
  };

  struct B : public A {
    B();
    ~B() __attribute__((noreturn));
  };

// CHECK-LABEL: int test1(int *x)
// CHECK: 1: 1
// CHECK-NEXT: 2: return
// CHECK-NEXT: ~NoReturnSingleSuccessor::B() (Implicit destructor)
// CHECK-NEXT: Preds (1)
// CHECK-NEXT: Succs (1): B0
  int test1(int *x) {
    B b;
    if (x)
      return 1;
  }

// CHECK-LABEL: int test2(int *x)
// CHECK: 1: 1
// CHECK-NEXT: 2: return
// CHECK-NEXT: destructor
// CHECK-NEXT: Preds (1)
// CHECK-NEXT: Succs (1): B0
  int test2(int *x) {
    const A& a = B();
    if (x)
      return 1;
  }
}

// Test CFG support for "extending" an enum.
// CHECK-LABEL: int test_enum_with_extension(enum MyEnum value)
// CHECK:  [B7 (ENTRY)]
// CHECK-NEXT:    Succs (1): B2
// CHECK:  [B1]
// CHECK-NEXT:    1: x
// CHECK-NEXT:    2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:    3: return [B1.2];
// CHECK-NEXT:    Preds (5): B3 B4 B5 B6 B2(Unreachable)
// CHECK-NEXT:    Succs (1): B0
// CHECK:  [B2]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: int x = 0;
// CHECK-NEXT:    3: value
// CHECK-NEXT:    4: [B2.3] (ImplicitCastExpr, LValueToRValue, enum MyEnum)
// CHECK-NEXT:    5: [B2.4] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:    T: switch [B2.5]
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (5): B3 B4 B5 B6 B1(Unreachable)
// CHECK:  [B3]
// CHECK-NEXT:   case D:
// CHECK-NEXT:    1: 4
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B3.2] = [B3.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B4]
// CHECK-NEXT:   case C:
// CHECK-NEXT:    1: 3
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B4.2] = [B4.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B5]
// CHECK-NEXT:   case B:
// CHECK-NEXT:    1: 2
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B5.2] = [B5.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B6]
// CHECK-NEXT:   case A:
// CHECK-NEXT:    1: 1
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B6.2] = [B6.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
enum MyEnum { A, B, C };
static const enum MyEnum D = (enum MyEnum) 32;

int test_enum_with_extension(enum MyEnum value) {
  int x = 0;
  switch (value) {
    case A: x = 1; break;
    case B: x = 2; break;
    case C: x = 3; break;
    case D: x = 4; break;
  }
  return x;
}

// CHECK-LABEL: int test_enum_with_extension_default(enum MyEnum value)
// CHECK:  [B7 (ENTRY)]
// CHECK-NEXT:    Succs (1): B2
// CHECK:  [B1]
// CHECK-NEXT:    1: x
// CHECK-NEXT:    2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:    3: return [B1.2];
// CHECK-NEXT:    Preds (4): B3 B4 B5 B6
// CHECK-NEXT:    Succs (1): B0
// CHECK:  [B2]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: int x = 0;
// CHECK-NEXT:    3: value
// CHECK-NEXT:    4: [B2.3] (ImplicitCastExpr, LValueToRValue, enum MyEnum)
// CHECK-NEXT:    5: [B2.4] (ImplicitCastExpr, IntegralCast, int)
// CHECK-NEXT:    T: switch [B2.5]
// CHECK-NEXT:    Preds (1): B7
// CHECK-NEXT:    Succs (4): B4 B5 B6 B3(Unreachable)
// CHECK:  [B3]
// CHECK-NEXT:   default:
// CHECK-NEXT:    1: 4
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B3.2] = [B3.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2(Unreachable)
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B4]
// CHECK-NEXT:   case C:
// CHECK-NEXT:    1: 3
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B4.2] = [B4.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B5]
// CHECK-NEXT:   case B:
// CHECK-NEXT:    1: 2
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B5.2] = [B5.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B6]
// CHECK-NEXT:   case A:
// CHECK-NEXT:    1: 1
// CHECK-NEXT:    2: x
// CHECK-NEXT:    3: [B6.2] = [B6.1]
// CHECK-NEXT:    T: break;
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1
int test_enum_with_extension_default(enum MyEnum value) {
  int x = 0;
  switch (value) {
    case A: x = 1; break;
    case B: x = 2; break;
    case C: x = 3; break;
    default: x = 4; break;
  }
  return x;
}

// CHECK-LABEL: void test_placement_new()
// CHECK:  [B2 (ENTRY)]
// CHECK-NEXT:  Succs (1): B1
// CHECK:  [B1]
// CHECK-NEXT:  1: int buffer[16];
// CHECK-NEXT:  2: buffer
// CHECK-NEXT:  3: [B1.2] (ImplicitCastExpr, ArrayToPointerDecay, int *)
// CHECK-NEXT:  4: [B1.3] (ImplicitCastExpr, BitCast, void *)
// CHECK-NEXT:  5: CFGNewAllocator(MyClass *)
// WARNINGS-NEXT:  6:  (CXXConstructExpr, class MyClass)
// ANALYZER-NEXT:  6:  (CXXConstructExpr, [B1.7], class MyClass)
// CHECK-NEXT:  7: new ([B1.4]) MyClass([B1.6])
// CHECK-NEXT:  8: MyClass *obj = new (buffer) MyClass();
// CHECK-NEXT:  Preds (1): B2
// CHECK-NEXT:  Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:  Preds (1): B1

extern void* operator new (unsigned long sz, void* v);
extern void* operator new[] (unsigned long sz, void* ptr);

class MyClass {
public:
  MyClass() {}
  ~MyClass() {}
};

void test_placement_new() {
  int buffer[16];
  MyClass* obj = new (buffer) MyClass();
}

// CHECK-LABEL: void test_placement_new_array()
// CHECK:  [B2 (ENTRY)]
// CHECK-NEXT:  Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:  1: int buffer[16];
// CHECK-NEXT:  2: buffer
// CHECK-NEXT:  3: [B1.2] (ImplicitCastExpr, ArrayToPointerDecay, int *)
// CHECK-NEXT:  4: [B1.3] (ImplicitCastExpr, BitCast, void *)
// CHECK-NEXT:  5: 5
// CHECK-NEXT:  6: CFGNewAllocator(MyClass *)
// WARNINGS-NEXT:  7:  (CXXConstructExpr, class MyClass [5])
// ANALYZER-NEXT:  7:  (CXXConstructExpr, [B1.8], class MyClass [5])
// CHECK-NEXT:  8: new ([B1.4]) MyClass {{\[\[}}B1.5]]
// CHECK-NEXT:  9: MyClass *obj = new (buffer) MyClass [5];
// CHECK-NEXT:  Preds (1): B2
// CHECK-NEXT:  Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:  Preds (1): B1

void test_placement_new_array() {
  int buffer[16];
  MyClass* obj = new (buffer) MyClass[5];
}


// CHECK-LABEL: void test_lifetime_extended_temporaries()
// CHECK: [B1]
struct LifetimeExtend { LifetimeExtend(int); ~LifetimeExtend(); };
struct Aggregate { const LifetimeExtend a; const LifetimeExtend b; };
struct AggregateRef { const LifetimeExtend &a; const LifetimeExtend &b; };
void test_lifetime_extended_temporaries() {
  // CHECK: LifetimeExtend(1);
  // CHECK-NEXT: : 1
  // CHECK-NEXT: ~LifetimeExtend()
  // CHECK-NOT: ~LifetimeExtend()
  {
    const LifetimeExtend &l = LifetimeExtend(1);
    1;
  }
  // CHECK: LifetimeExtend(2)
  // CHECK-NEXT: ~LifetimeExtend()
  // CHECK-NEXT: : 2
  // CHECK-NOT: ~LifetimeExtend()
  {
    // No life-time extension.
    const int &l = (LifetimeExtend(2), 2);
    2;
  }
  // CHECK: LifetimeExtend(3)
  // CHECK-NEXT: : 3
  // CHECK-NEXT: ~LifetimeExtend()
  // CHECK-NOT: ~LifetimeExtend()
  {
    // The last one is lifetime extended.
    const LifetimeExtend &l = (3, LifetimeExtend(3));
    3;
  }
  // CHECK: LifetimeExtend(4)
  // CHECK-NEXT: ~LifetimeExtend()
  // CHECK-NEXT: ~LifetimeExtend()
  // CHECK-NEXT: : 4
  // CHECK-NOT: ~LifetimeExtend()
  {
    Aggregate a{LifetimeExtend(4), LifetimeExtend(4)};
    4;
  }
  // CHECK: LifetimeExtend(5)
  // CHECK-NEXT: : 5
  // FIXME: We want to emit the destructors of the lifetime
  // extended variables here.
  // CHECK-NOT: ~LifetimeExtend()
  {
    AggregateRef a{LifetimeExtend(5), LifetimeExtend(5)};
    5;
  }
  // FIXME: Add tests for lifetime extension via subobject
  // references (LifetimeExtend().some_member).
}


// FIXME: The destructor for 'a' shouldn't be there because it's deleted
// in the union.
// CHECK-LABEL: void foo()
// CHECK:  [B2 (ENTRY)]
// CHECK-NEXT:    Succs (1): B1
// CHECK:  [B1]
// WARNINGS-NEXT:    1:  (CXXConstructExpr, struct pr37688_deleted_union_destructor::A)
// ANALYZER-NEXT:    1:  (CXXConstructExpr, [B1.2], struct pr37688_deleted_union_destructor::A)
// CHECK-NEXT:    2: pr37688_deleted_union_destructor::A a;
// CHECK-NEXT:    3: [B1.2].~pr37688_deleted_union_destructor::A() (Implicit destructor)
// CHECK-NEXT:    Preds (1): B2
// CHECK-NEXT:    Succs (1): B0
// CHECK:  [B0 (EXIT)]
// CHECK-NEXT:    Preds (1): B1

namespace pr37688_deleted_union_destructor {
struct S { ~S(); };
struct A {
  ~A() noexcept {}
  union {
    struct {
      S s;
    } ss;
  };
};
void foo() {
  A a;
}
} // end namespace pr37688_deleted_union_destructor


namespace return_statement_expression {
int unknown();

// CHECK-LABEL: int foo()
// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:    Succs (1): B5
// CHECK:       [B1]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: return [B1.1];
// CHECK-NEXT:    Preds (1): B5
// CHECK-NEXT:    Succs (1): B0
// CHECK:       [B2]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: ({ ... ; [B2.1] })
// CHECK-NEXT:    3: return [B2.2];
// CHECK-NEXT:    Preds (1): B4
// CHECK-NEXT:    Succs (1): B0
// FIXME: Why do we have [B3] at all?
// CHECK:       [B3]
// CHECK-NEXT:    Succs (1): B4
// CHECK:       [B4]
// CHECK-NEXT:    1: 0
// CHECK-NEXT:    2: [B4.1] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: while [B4.2]
// CHECK-NEXT:    Preds (2): B3 B5
// CHECK-NEXT:    Succs (2): NULL B2
// CHECK:       [B5]
// CHECK-NEXT:    1: unknown
// CHECK-NEXT:    2: [B5.1] (ImplicitCastExpr, FunctionToPointerDecay, int (*)(void))
// CHECK-NEXT:    3: [B5.2]()
// CHECK-NEXT:    4: [B5.3] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:    T: if [B5.4]
// CHECK-NEXT:    Preds (1): B6
// CHECK-NEXT:    Succs (2): B4 B1
// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:    Preds (2): B1 B2
int foo() {
  if (unknown())
    return ({
      while (0)
        ;
      0;
    });
  else
    return 0;
}
} // namespace statement_expression_in_return

// CHECK-LABEL: void vla_simple(int x)
// CHECK: [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: int vla[x];
void vla_simple(int x) {
  int vla[x];
}

// CHECK-LABEL: void vla_typedef(int x)
// CHECK: [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: typedef int VLA[x];
void vla_typedef(int x) {
  typedef int VLA[x];
}

// CHECK-LABEL: void vla_typealias(int x)
// CHECK: [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: using VLA = int [x];
void vla_typealias(int x) {
  using VLA = int[x];
}

// CHECK-LABEL: void vla_typedef_multi(int x, int y)
// CHECK:  [B1]
// CHECK-NEXT:   1: y
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: x
// CHECK-NEXT:   4: [B1.3] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   5: typedef int VLA[x][y];
void vla_typedef_multi(int x, int y) {
  typedef int VLA[x][y];
}

// CHECK-LABEL: void vla_typedefname_multi(int x, int y)
// CHECK:  [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: typedef int VLA[x];
// CHECK-NEXT:   4: y
// CHECK-NEXT:   5: [B1.4] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   6: typedef VLA VLA1[y];
// CHECK-NEXT:   7: 3
// CHECK-NEXT:   8: using VLA2 = VLA1 [3];
// CHECK-NEXT:   9: 4
// CHECK-NEXT:  10: VLA2 vla[4];
void vla_typedefname_multi(int x, int y) {
  typedef int VLA[x];
  typedef VLA VLA1[y];
  using VLA2 = VLA1[3];
  VLA2 vla[4];
}

// CHECK-LABEL: int vla_evaluate(int x)
// CHECK:  [B1]
// CHECK-NEXT:   1: x
// CHECK-NEXT:   2: ++[B1.1]
// CHECK-NEXT:   3: [B1.2] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   4: typedef int VLA[++x];
// CHECK-NEXT:   5: x
// CHECK-NEXT:   6: ++[B1.5]
// CHECK-NEXT:   7: [B1.6] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   8: sizeof(int [++x])
// CHECK-NEXT:   9: alignof(int [++x])
// CHECK-NEXT:  10: 0
// CHECK-NEXT:  11: x
// CHECK-NEXT:  12: [B1.11] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  13: return [B1.12];
int vla_evaluate(int x) {
  // Evaluates the ++x
  typedef int VLA[++x];
  sizeof(int[++x]);

  // Do not evaluate the ++x
  _Alignof(int[++x]);
  _Generic((int(*)[++x])0, default : 0);

  return x;
}

// CHECK-LABEL: void CommaTemp::f()
// CHECK:       [B1]
// CHECK-NEXT:    1: CommaTemp::A() (CXXConstructExpr,
// CHECK-NEXT:    2: [B1.1] (BindTemporary)
// CHECK-NEXT:    3: CommaTemp::B() (CXXConstructExpr,
// CHECK-NEXT:    4: [B1.3] (BindTemporary)
// CHECK-NEXT:    5: ... , [B1.4]
// CHECK-NEXT:    6: ~CommaTemp::B() (Temporary object destructor)
// CHECK-NEXT:    7: ~CommaTemp::A() (Temporary object destructor)
namespace CommaTemp {
  struct A { ~A(); };
  struct B { ~B(); };
  void f();
}
void CommaTemp::f() {
  A(), B();
}

// CHECK-LABEL: template<> int *PR18472<int>()
// CHECK: [B2 (ENTRY)]
// CHECK-NEXT:   Succs (1): B1
// CHECK: [B1]
// CHECK-NEXT:   1: 0
// CHECK-NEXT:   2: [B1.1] (ImplicitCastExpr, NullToPointer, PR18472_t)
// CHECK-NEXT:   3: (PR18472_t)[B1.2] (CStyleCastExpr, NoOp, PR18472_t)
// CHECK-NEXT:   4: CFGNewAllocator(int *)
// CHECK-NEXT:   5: new (([B1.3])) int
// CHECK-NEXT:   6: return [B1.5];
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1

extern "C" typedef int *PR18472_t;
void *operator new (unsigned long, PR18472_t);
template <class T> T *PR18472() {
  return new (((PR18472_t) 0)) T;
}
void PR18472_helper() {
  PR18472<int>();
}
