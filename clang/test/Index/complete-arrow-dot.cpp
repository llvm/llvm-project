struct X {
  void doSomething();

  int field;
};

void X::doSomething() {
  // RUN: c-index-test -code-completion-at=%s:10:8 %s | FileCheck %s
  // RUN: env CINDEXTEST_COMPLETION_INCLUDE_FIXITS=1 c-index-test -code-completion-at=%s:10:8 %s | FileCheck -check-prefix=CHECK-WITH-CORRECTION %s
  this.;
}

void doSomething() {
  // RUN: c-index-test -code-completion-at=%s:17:6 %s | FileCheck -check-prefix=CHECK-ARROW-TO-DOT %s
  // RUN: env CINDEXTEST_COMPLETION_INCLUDE_FIXITS=1 c-index-test -code-completion-at=%s:17:6 %s | FileCheck -check-prefix=CHECK-ARROW-TO-DOT-WITH-CORRECTION %s
  X x;
  x->
}

// CHECK-NOT: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34) (requires fix-it:{{.*}}
// CHECK-NOT: FieldDecl:{ResultType int}{TypedText field} (35) (requires fix-it:{{.*}}
// CHECK-NOT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-NOT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder X &&}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-NOT: StructDecl:{TypedText X}{Text ::} (75) (requires fix-it:{{.*}}
// CHECK-NOT: CXXDestructor:{ResultType void}{TypedText ~X}{LeftParen (}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK: Completion contexts:
// CHECK-NEXT: Dot member access

// CHECK-WITH-CORRECTION: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34) (requires fix-it:{{.*}}
// CHECK-WITH-CORRECTION-NEXT: FieldDecl:{ResultType int}{TypedText field} (35) (requires fix-it:{{.*}}
// CHECK-WITH-CORRECTION-NEXT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-WITH-CORRECTION-NEXT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder X &&}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-WITH-CORRECTION-NEXT: StructDecl:{TypedText X}{Text ::} (75) (requires fix-it:{{.*}}
// CHECK-WITH-CORRECTION-NEXT: CXXDestructor:{ResultType void}{TypedText ~X}{LeftParen (}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-WITH-CORRECTION-NEXT: Completion contexts:
// CHECK-WITH-CORRECTION-NEXT: Dot member access

// CHECK-ARROW-TO-DOT-NOT: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-NOT: FieldDecl:{ResultType int}{TypedText field} (35) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-NOT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-NOT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder X &&}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-NOT: StructDecl:{TypedText X}{Text ::} (75) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-NOT: CXXDestructor:{ResultType void}{TypedText ~X}{LeftParen (}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT: Completion contexts:
// CHECK-ARROW-TO-DOT-NEXT: Unknown

// CHECK-ARROW-TO-DOT-WITH-CORRECTION: CXXMethod:{ResultType void}{TypedText doSomething}{LeftParen (}{RightParen )} (34) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: FieldDecl:{ResultType int}{TypedText field} (35) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: CXXMethod:{ResultType X &}{TypedText operator=}{LeftParen (}{Placeholder X &&}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: StructDecl:{TypedText X}{Text ::} (75) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: CXXDestructor:{ResultType void}{TypedText ~X}{LeftParen (}{RightParen )} (79) (requires fix-it:{{.*}}
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: Completion contexts:
// CHECK-ARROW-TO-DOT-WITH-CORRECTION-NEXT: Arrow member access
