struct Base1 {
  int member1;
  float member2;
};

struct Base2 {
  int member1;
  double member3;
  void memfun1(int);
};

struct Base3 : Base1, Base2 {
  void memfun1(float);
  void memfun1(double) const;
  void memfun2(int);
};

struct Derived : Base3 {
  template <typename T> Derived(T);
  Derived(int);
  int member4;
  int memfun3(int);
};

class Proxy {
public:
  Derived *operator->() const;
};

void test(const Proxy &p) {
  p->
}

struct Test1 {
  Base1 b;

  static void sfunc() {
    b. // expected-error {{invalid use of member 'b' in static member function}}
  }
};

struct Foo {
  void foo() const;
  static void foo(bool);
};

struct Bar {
  void foo(bool param) {
    Foo::foo(  );// unresolved member expression with an implicit base
  }
};

  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:31:6 %s -o - | FileCheck -check-prefix=CHECK-CC1 --implicit-check-not="Derived : Derived(" %s
  // CHECK-CC1: Base1 (InBase) : Base1::
  // CHECK-CC1: member1 (InBase) : [#int#][#Base1::#]member1
  // CHECK-CC1: member1 (InBase) : [#int#][#Base2::#]member1
  // CHECK-CC1: member2 (InBase) : [#float#][#Base1::#]member2
  // CHECK-CC1: member3 (InBase)
  // CHECK-CC1: member4
  // CHECK-CC1: memfun1 (InBase) : [#void#][#Base3::#]memfun1(<#float#>)
  // CHECK-CC1: memfun1 (InBase) : [#void#][#Base3::#]memfun1(<#double#>)[# const#]
  // CHECK-CC1: memfun1 (Hidden,InBase) : [#void#]Base2::memfun1(<#int#>)
  // CHECK-CC1: memfun2 (InBase) : [#void#][#Base3::#]memfun2(<#int#>)
  // CHECK-CC1: memfun3 : [#int#]memfun3(<#int#>)

// Make sure this doesn't crash
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:38:7 %s -verify

// Make sure this also doesn't crash
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:49:14 %s


template<typename T>
class BaseTemplate {
public:
  T baseTemplateFunction();

  T baseTemplateField;
};

template<typename T, typename S>
class TemplateClass: public Base1 , public BaseTemplate<T> {
public:
  T function() { }
  T field;

  TemplateClass<S, T> &relatedField;
  BaseTemplate<S> &relatedFunction();

  void overload1(const T &);
  void overload1(const S &);
};

template<typename T, typename S>
void completeDependentMembers(TemplateClass<T, S> &object,
                              TemplateClass<int, S> *object2) {
  object.field;
  object2->field;
// CHECK-CC2: baseTemplateField (InBase) : [#T#][#BaseTemplate<T>::#]baseTemplateField
// CHECK-CC2: baseTemplateFunction (InBase) : [#T#][#BaseTemplate<T>::#]baseTemplateFunction()
// CHECK-CC2: field : [#T#]field
// CHECK-CC2: function : [#T#]function()
// CHECK-CC2: member1 (InBase) : [#int#][#Base1::#]member1
// CHECK-CC2: member2 (InBase) : [#float#][#Base1::#]member2
// CHECK-CC2: overload1 : [#void#]overload1(<#const T &#>)
// CHECK-CC2: overload1 : [#void#]overload1(<#const S &#>)

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:97:10 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:98:12 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s

  auto copy_object = object;
  auto copy_object2 = object2;
  object.field;
  object2->field;
// CHECK-AUTO: field : [#T#]field
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:113:10 %s -o - | FileCheck -check-prefix=CHECK-AUTO %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:114:12 %s -o - | FileCheck -check-prefix=CHECK-AUTO %s

  object.relatedField.relatedFunction().baseTemplateField;
// CHECK-DEP-CHAIN: baseTemplateField : [#T#]baseTemplateField
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:119:41 %s -o - | FileCheck -check-prefix=CHECK-DEP-CHAIN %s
}


void completeDependentSpecializedMembers(TemplateClass<int, double> &object,
                                         TemplateClass<int, double> *object2) {
  object.field;
  object2->field;
// CHECK-CC3: baseTemplateField (InBase) : [#int#][#BaseTemplate<int>::#]baseTemplateField
// CHECK-CC3: baseTemplateFunction (InBase) : [#int#][#BaseTemplate<int>::#]baseTemplateFunction()
// CHECK-CC3: field : [#int#]field
// CHECK-CC3: function : [#int#]function()
// CHECK-CC3: member1 (InBase) : [#int#][#Base1::#]member1
// CHECK-CC3: member2 (InBase) : [#float#][#Base1::#]member2
// CHECK-CC3: overload1 : [#void#]overload1(<#const int &#>)
// CHECK-CC3: overload1 : [#void#]overload1(<#const double &#>)

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:127:10 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:128:12 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
}

template <typename T>
class Template {
public:
  BaseTemplate<int> o1;
  BaseTemplate<T> o2;

  void function() {
    o1.baseTemplateField;
// CHECK-CC4: BaseTemplate : BaseTemplate::
// CHECK-CC4: baseTemplateField : [#int#]baseTemplateField
// CHECK-CC4: baseTemplateFunction : [#int#]baseTemplateFunction()
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:149:8 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
    o2.baseTemplateField;
// CHECK-CC5: BaseTemplate : BaseTemplate::
// CHECK-CC5: baseTemplateField : [#T#]baseTemplateField
// CHECK-CC5: baseTemplateFunction : [#T#]baseTemplateFunction()
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:154:8 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
    this->o1;
// CHECK-CC6: [#void#]function()
// CHECK-CC6: o1 : [#BaseTemplate<int>#]o1
// CHECK-CC6: o2 : [#BaseTemplate<T>#]o2
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:159:11 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  }

  static void staticFn(T &obj);

  struct Nested { };
};

template<typename T>
void dependentColonColonCompletion() {
  Template<T>::staticFn();
// CHECK-CC7: function : [#void#]function()
// CHECK-CC7: Nested : Nested
// CHECK-CC7: o1 : [#BaseTemplate<int>#]o1
// CHECK-CC7: o2 : [#BaseTemplate<T>#]o2
// CHECK-CC7: staticFn : [#void#]staticFn(<#T &obj#>)
// CHECK-CC7: Template : Template
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:173:16 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
  typename Template<T>::Nested m;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:181:25 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
}

class Proxy2 {
public:
  Derived *operator->() const;
  int member5;
};

void test2(const Proxy2 &p) {
  p->
}

void test3(const Proxy2 &p) {
  p.
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:192:6 %s -o - | FileCheck -check-prefix=CHECK-CC8 --implicit-check-not="Derived : Derived(" %s
// CHECK-CC8: Base1 (InBase) : Base1::
// CHECK-CC8: member1 (InBase) : [#int#][#Base1::#]member1
// CHECK-CC8: member1 (InBase) : [#int#][#Base2::#]member1
// CHECK-CC8: member2 (InBase) : [#float#][#Base1::#]member2
// CHECK-CC8: member3 (InBase) : [#double#][#Base2::#]member3
// CHECK-CC8: member4 : [#int#]member4
// CHECK-CC8: member5 : [#int#]member5 (requires fix-it: {192:4-192:6} to ".")
// CHECK-CC8: memfun1 (InBase) : [#void#][#Base3::#]memfun1(<#float#>)
// CHECK-CC8: memfun1 (InBase) : [#void#][#Base3::#]memfun1(<#double#>)[# const#]
// CHECK-CC8: memfun1 (Hidden,InBase) : [#void#]Base2::memfun1(<#int#>)
// CHECK-CC8: memfun2 (InBase) : [#void#][#Base3::#]memfun2(<#int#>)
// CHECK-CC8: memfun3 : [#int#]memfun3(<#int#>)
// CHECK-CC8: operator-> : [#Derived *#]operator->()[# const#] (requires fix-it: {192:4-192:6} to ".")

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:196:6 %s -o - | FileCheck -check-prefix=CHECK-CC9 --implicit-check-not="Derived : Derived(" %s
// CHECK-CC9: Base1 (InBase) : Base1::
// CHECK-CC9: member1 (InBase) : [#int#][#Base1::#]member1 (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: member1 (InBase) : [#int#][#Base2::#]member1 (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: member2 (InBase) : [#float#][#Base1::#]member2 (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: member3 (InBase) : [#double#][#Base2::#]member3 (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: member4 : [#int#]member4 (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: member5 : [#int#]member5
// CHECK-CC9: memfun1 (InBase) : [#void#][#Base3::#]memfun1(<#float#>) (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: memfun1 (InBase) : [#void#][#Base3::#]memfun1(<#double#>)[# const#] (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: memfun1 (Hidden,InBase) : [#void#]Base2::memfun1(<#int#>) (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: memfun2 (InBase) : [#void#][#Base3::#]memfun2(<#int#>) (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: memfun3 : [#int#]memfun3(<#int#>) (requires fix-it: {196:4-196:5} to "->")
// CHECK-CC9: operator-> : [#Derived *#]operator->()[# const#]

// These overload sets differ only by return type and this-qualifiers.
// So for any given callsite, only one is available.
struct Overloads {
  double ConstOverload(char);
  int ConstOverload(char) const;

  int RefOverload(char) &;
  double RefOverload(char) const&;
  char RefOverload(char) &&;
};
void testLValue(Overloads& Ref) {
  Ref.
}
void testConstLValue(const Overloads& ConstRef) {
  ConstRef.
}
void testRValue() {
  Overloads().
}
void testXValue(Overloads& X) {
  static_cast<Overloads&&>(X).
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:240:7 %s -o - | FileCheck -check-prefix=CHECK-LVALUE %s \
// RUN: --implicit-check-not="[#int#]ConstOverload(" \
// RUN: --implicit-check-not="[#double#]RefOverload(" \
// RUN: --implicit-check-not="[#char#]RefOverload("
// CHECK-LVALUE-DAG: [#double#]ConstOverload(
// CHECK-LVALUE-DAG: [#int#]RefOverload(

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:243:12 %s -o - | FileCheck -check-prefix=CHECK-CONSTLVALUE %s \
// RUN: --implicit-check-not="[#double#]ConstOverload(" \
// RUN: --implicit-check-not="[#int#]RefOverload(" \
// RUN: --implicit-check-not="[#char#]RefOverload("
// CHECK-CONSTLVALUE: [#int#]ConstOverload(
// CHECK-CONSTLVALUE: [#double#]RefOverload(

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:246:15 %s -o - | FileCheck -check-prefix=CHECK-PRVALUE %s \
// RUN: --implicit-check-not="[#int#]ConstOverload(" \
// RUN: --implicit-check-not="[#int#]RefOverload(" \
// RUN: --implicit-check-not="[#double#]RefOverload("
// CHECK-PRVALUE: [#double#]ConstOverload(
// CHECK-PRVALUE: [#char#]RefOverload(

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:249:31 %s -o - | FileCheck -check-prefix=CHECK-XVALUE %s \
// RUN: --implicit-check-not="[#int#]ConstOverload(" \
// RUN: --implicit-check-not="[#int#]RefOverload(" \
// RUN: --implicit-check-not="[#double#]RefOverload("
// CHECK-XVALUE: [#double#]ConstOverload(
// CHECK-XVALUE: [#char#]RefOverload(

void testOverloadOperator() {
  struct S {
    char operator=(int) const;
    int operator=(int);
  } s;
  return s.
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:285:12 %s -o - | FileCheck -check-prefix=CHECK-OPER %s \
// RUN: --implicit-check-not="[#char#]operator=("
// CHECK-OPER: [#int#]operator=(

struct S { int member; };
S overloaded(int);
S overloaded(double);
void foo() {
  // No overload matches, but we have recovery-expr with the correct type.
  overloaded().
}
// RUN: not %clang_cc1 -fsyntax-only -frecovery-ast -frecovery-ast-type -code-completion-at=%s:296:16 %s -o - | FileCheck -check-prefix=CHECK-RECOVERY %s
// CHECK-RECOVERY: [#int#]member
template <typename T>
void fooDependent(T t) {
  // Overload not resolved, but we notice all candidates return the same type.
  overloaded(t).
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:303:17 %s -o - | FileCheck -check-prefix=CHECK-OVERLOAD %s
// CHECK-OVERLOAD: [#int#]member

struct Base4 {
  Base4 base4();
};

template <typename T>
struct Derived2 : Base4 {};

template <typename T>
void testMembersFromBasesInDependentContext() {
  Derived2<T> X;
  (void)X.base4().base4();
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:318:19 %s -o - | FileCheck -check-prefix=CHECK-MEMBERS-FROM-BASE-DEPENDENT %s
  // CHECK-MEMBERS-FROM-BASE-DEPENDENT: [#Base4#]base4
}

namespace members_using_fixits {
  struct Bar {
    void method();
    int field;
  };
  struct Baz: Bar {
    using Bar::method;
    using Bar::field;
  };
  void testMethod(Baz* ptr) {
    ptr.m
  }
  // RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:333:10 %s -o - | FileCheck -check-prefix=CHECK-METHOD-DECLARED-VIA-USING %s
  // CHECK-METHOD-DECLARED-VIA-USING: [#void#]method() (requires fix-it: {333:8-333:9} to "->")

  void testField(Baz* ptr) {
    ptr.f
  }
  // RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:339:10 %s -o - | FileCheck -check-prefix=CHECK-FIELD-DECLARED-VIA-USING %s
  // CHECK-FIELD-DECLARED-VIA-USING: [#int#]field (requires fix-it: {339:8-339:9} to "->")
}

namespace function_can_be_call {
  struct S {
    template <typename T, typename U, typename V = int>
    T foo(U, V);
  };

  void test() {
    &S::f
  }
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:352:9 %s -o - | FileCheck -check-prefix=CHECK_FUNCTION_CAN_BE_CALL %s
  // CHECK_FUNCTION_CAN_BE_CALL: COMPLETION: foo : [#T#]foo<<#typename T#>, <#typename U#>>(<#U#>, <#V#>)
}

namespace deref_dependent_this {
template <typename T>
class A {
  int field;

  void function() {
    (*this).field;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:364:13 %s -o - | FileCheck -check-prefix=CHECK-DEREF-THIS %s
// CHECK-DEREF-THIS: field : [#int#]field
// CHECK-DEREF-THIS: [#void#]function()
  }
};

template <typename Element>
struct RepeatedField {
  void Add();
};

template <typename T>
RepeatedField<T>* MutableRepeatedField() {}

template <class T>
void Foo() {
  auto& C = *MutableRepeatedField<T>();
  C.
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:382:5 %s -o - | FileCheck -check-prefix=CHECK-DEREF-DEPENDENT %s
// CHECK-DEREF-DEPENDENT: [#void#]Add()
}
