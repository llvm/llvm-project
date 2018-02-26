class X {
  void doSomething();

  int field __attribute((annotate("one"), annotate("two"), annotate("three")));

public __attribute__((annotate("some annotation"))):
  void func2();
  int member2 __attribute__((annotate("another annotation")));
};

void X::doSomething() {
  // RUN: c-index-test -code-completion-at=%s:13:9 %s | FileCheck %s
  this->;
}

// CHECK: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34)
// CHECK: FieldDecl:{ResultType int}{TypedText field} (35) ("three", "two", "one")
// CHECK: CXXMethod:{ResultType void}{TypedText func2}{LeftParen (}{RightParen )} (34) ("some annotation")
// CHECK: FieldDecl:{ResultType int}{TypedText member2} (35) ("another annotation", "some annotation")
// CHECK: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )} (79)
// CHECK: ClassDecl:{TypedText X}{Text ::} (75)
// CHECK: CXXDestructor:{ResultType void}{TypedText ~X}{LeftParen (}{RightParen )} (79)

