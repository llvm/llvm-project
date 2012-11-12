// Note: the run lines follow their respective tests, since line/column
// matter in this test.


void f(int x) {
  if (x) {
  } 
}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:7:4 %s | FileCheck -check-prefix=CHECK-IF-ELSE %s
// CHECK-IF-ELSE: NotImplemented:{TypedText else}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Placeholder statements}{VerticalSpace  }{RightBrace }} (40)
// CHECK-IF-ELSE: NotImplemented:{TypedText else}{HorizontalSpace  }{Text if}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Placeholder statements}{VerticalSpace  }{RightBrace }} (40)

// RUN: c-index-test -code-completion-at=%s:7:4 %s | FileCheck -check-prefix=CHECK-IF-ELSE-SIMPLE %s
// CHECK-IF-ELSE-SIMPLE: NotImplemented:{TypedText else} (40)
// CHECK-IF-ELSE-SIMPLE: NotImplemented:{TypedText else}{HorizontalSpace  }{Text if}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )} (40)
