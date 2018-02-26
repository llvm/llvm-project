namespace {
class MyCls {
  void in_foo() {
    vec.x = 0;
  }
  void out_foo();

  struct Vec { int x, y; };
  Vec vec;
};

void MyCls::out_foo() {
  vec.x = 0;
}

class OtherClass : public MyCls {
public:
  OtherClass(const OtherClass &other) : MyCls(other), value(value) { }

private:
  int value;
  MyCls *object;
};
}

// RUN: c-index-test -code-completion-at=%s:4:9 -std=c++98 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:13:7 -std=c++98 %s | FileCheck %s
// CHECK:      CXXMethod:{ResultType Vec &}{TypedText operator=}{LeftParen (}{Placeholder const Vec &}{RightParen )} (79)
// CHECK-NEXT: StructDecl:{TypedText Vec}{Text ::} (75)
// CHECK-NEXT: FieldDecl:{ResultType int}{TypedText x} (35)
// CHECK-NEXT: FieldDecl:{ResultType int}{TypedText y} (35)
// CHECK-NEXT: CXXDestructor:{ResultType void}{TypedText ~Vec}{LeftParen (}{RightParen )} (79)
// CHECK-NEXT: Completion contexts:
// CHECK-NEXT: Dot member access
// CHECK-NEXT: Container Kind: StructDecl

// RUN: c-index-test -code-completion-at=%s:18:41 %s | FileCheck -check-prefix=CHECK-CTOR-INIT %s
// CHECK-CTOR-INIT: NotImplemented:{TypedText MyCls}{LeftParen (}{Placeholder args}{RightParen )} (7)
// CHECK-CTOR-INIT: MemberRef:{TypedText object}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CTOR-INIT: MemberRef:{TypedText value}{LeftParen (}{Placeholder args}{RightParen )} (35)
// RUN: c-index-test -code-completion-at=%s:18:55 %s | FileCheck -check-prefix=CHECK-CTOR-INIT-2 %s
// CHECK-CTOR-INIT-2-NOT: NotImplemented:{TypedText MyCls}{LeftParen (}{Placeholder args}{RightParen )}
// CHECK-CTOR-INIT-2: MemberRef:{TypedText object}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CTOR-INIT-2: MemberRef:{TypedText value}{LeftParen (}{Placeholder args}{RightParen )} (7)
