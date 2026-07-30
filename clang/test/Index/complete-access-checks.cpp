struct X {
  int member1;
  void func1();
protected:
  int member2;
  void func2();
private:
  int member3;
  void func3();
};

struct Y: protected X {
  void doSomething();
};

class Z {
public:
  int member1;
  void func1();
protected:
  int member2;
  void func2();
private:
  int member3;
  void func3();
};

void Y::doSomething() {
  // RUN: c-index-test -code-completion-at=%s:30:9 %s | FileCheck -check-prefix=CHECK-SUPER-ACCESS %s
  this->;

  // RUN: c-index-test -code-completion-at=%s:33:3 %s | FileCheck -check-prefix=CHECK-SUPER-ACCESS-IMPLICIT %s
  

  Z that;
  // RUN: c-index-test -code-completion-at=%s:37:8 %s | FileCheck -check-prefix=CHECK-ACCESS %s
  that.
}

// CHECK-SUPER-ACCESS: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34)
// CHECK-SUPER-ACCESS: CXXMethod:{ResultType void}{Informative X::}{TypedText func1}{LeftParen (}{RightParen )} (36)
// CHECK-SUPER-ACCESS: CXXMethod:{ResultType void}{Informative X::}{TypedText func2}{LeftParen (}{RightParen )} (36){{$}}
// CHECK-SUPER-ACCESS: CXXMethod:{ResultType void}{Informative X::}{TypedText func3}{LeftParen (}{RightParen )} (36) (inaccessible)
// CHECK-SUPER-ACCESS: FieldDecl:{ResultType int}{Informative X::}{TypedText member1} (37)
// CHECK-SUPER-ACCESS: FieldDecl:{ResultType int}{Informative X::}{TypedText member2} (37){{$}}
// CHECK-SUPER-ACCESS: FieldDecl:{ResultType int}{Informative X::}{TypedText member3} (37) (inaccessible)
// CHECK-SUPER-ACCESS: CXXMethod:{ResultType Y &}{TypedText operator=}{LeftParen (}{Placeholder const Y &}{RightParen )} (79)
// CHECK-SUPER-ACCESS: CXXMethod:{ResultType X &}{Text X::}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )} (81)
// CHECK-SUPER-ACCESS: StructDecl:{TypedText X}{Text ::} (77)
// CHECK-SUPER-ACCESS: StructDecl:{TypedText Y}{Text ::} (75)
// CHECK-SUPER-ACCESS: CXXDestructor:{ResultType void}{Informative X::}{TypedText ~X}{LeftParen (}{RightParen )} (81)
// CHECK-SUPER-ACCESS: CXXDestructor:{ResultType void}{TypedText ~Y}{LeftParen (}{RightParen )} (79)

// CHECK-SUPER-ACCESS-IMPLICIT: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34)
// CHECK-SUPER-ACCESS-IMPLICIT: CXXMethod:{ResultType void}{TypedText func1}{LeftParen (}{RightParen )} (36)
// CHECK-SUPER-ACCESS-IMPLICIT: CXXMethod:{ResultType void}{TypedText func2}{LeftParen (}{RightParen )} (36){{$}}
// CHECK-SUPER-ACCESS-IMPLICIT: CXXMethod:{ResultType void}{TypedText func3}{LeftParen (}{RightParen )} (36) (inaccessible)
// CHECK-SUPER-ACCESS-IMPLICIT: FieldDecl:{ResultType int}{TypedText member1} (37)
// CHECK-SUPER-ACCESS-IMPLICIT: FieldDecl:{ResultType int}{TypedText member2} (37){{$}}
// CHECK-SUPER-ACCESS-IMPLICIT: FieldDecl:{ResultType int}{TypedText member3} (37) (inaccessible)

// CHECK-ACCESS: CXXMethod:{ResultType void}{TypedText func1}{LeftParen (}{RightParen )} (34)
// CHECK-ACCESS: CXXMethod:{ResultType void}{TypedText func2}{LeftParen (}{RightParen )} (34) (inaccessible)
// CHECK-ACCESS: CXXMethod:{ResultType void}{TypedText func3}{LeftParen (}{RightParen )} (34) (inaccessible)
// CHECK-ACCESS: FieldDecl:{ResultType int}{TypedText member1} (35)
// CHECK-ACCESS: FieldDecl:{ResultType int}{TypedText member2} (35) (inaccessible)
// CHECK-ACCESS: FieldDecl:{ResultType int}{TypedText member3} (35) (inaccessible)
// CHECK-ACCESS: CXXMethod:{ResultType Z &}{TypedText operator=}{LeftParen (}{Placeholder const Z &}{RightParen )} (79)
// CHECK-ACCESS: ClassDecl:{TypedText Z}{Text ::} (75)
// CHECK-ACCESS: CXXDestructor:{ResultType void}{TypedText ~Z}{LeftParen (}{RightParen )} (79)

class P {
protected:
  int member;
};

class Q : public P {
public:
  using P::member;
};

void f(P x, Q y) {
  // RUN: c-index-test -code-completion-at=%s:84:5 %s | FileCheck -check-prefix=CHECK-USING-INACCESSIBLE %s
  x.; // member is inaccessible
  // RUN: c-index-test -code-completion-at=%s:86:5 %s | FileCheck -check-prefix=CHECK-USING-ACCESSIBLE %s
  y.; // member is accessible
}

// CHECK-USING-INACCESSIBLE: FieldDecl:{ResultType int}{TypedText member} (35) (inaccessible)
// CHECK-USING-INACCESSIBLE: CXXMethod:{ResultType P &}{TypedText operator=}{LeftParen (}{Placeholder const P &}{RightParen )} (79)
// CHECK-USING-INACCESSIBLE: ClassDecl:{TypedText P}{Text ::} (75)
// CHECK-USING-INACCESSIBLE: CXXDestructor:{ResultType void}{TypedText ~P}{LeftParen (}{RightParen )} (79)

// CHECK-USING-ACCESSIBLE: FieldDecl:{ResultType int}{TypedText member} (35)
// CHECK-USING-ACCESSIBLE: CXXMethod:{ResultType Q &}{TypedText operator=}{LeftParen (}{Placeholder const Q &}{RightParen )} (79)
// CHECK-USING-ACCESSIBLE: CXXMethod:{ResultType P &}{Text P::}{TypedText operator=}{LeftParen (}{Placeholder const P &}{RightParen )} (81)
// CHECK-USING-ACCESSIBLE: ClassDecl:{TypedText P}{Text ::} (77)
// CHECK-USING-ACCESSIBLE: ClassDecl:{TypedText Q}{Text ::} (75)
// CHECK-USING-ACCESSIBLE: CXXDestructor:{ResultType void}{Informative P::}{TypedText ~P}{LeftParen (}{RightParen )} (81)
// CHECK-USING-ACCESSIBLE: CXXDestructor:{ResultType void}{TypedText ~Q}{LeftParen (}{RightParen )} (79)

class B {
protected:
  int member;
};

class C : private B {};


class D : public C {
 public:
  void f(::B *b);
};

void D::f(::B *that) {
  // RUN: c-index-test -code-completion-at=%s:117:9 %s | FileCheck -check-prefix=CHECK-PRIVATE-SUPER-THIS %s
  this->;
// CHECK-PRIVATE-SUPER-THIS: FieldDecl:{ResultType int}{Informative B::}{TypedText member} (37) (inaccessible)

  // RUN: c-index-test -code-completion-at=%s:121:9 %s | FileCheck -check-prefix=CHECK-PRIVATE-SUPER-THAT %s
  that->;
// CHECK-PRIVATE-SUPER-THAT: FieldDecl:{ResultType int}{TypedText member} (35) (inaccessible)
}
