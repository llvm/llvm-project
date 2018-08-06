// The line and column layout of this test is significant. Run lines
// are at the end.

#if 1
#endif

#define FOO(a, b) a##b
#define BAR
#ifdef FOO
#endif
#if defined(FOO)
#endif

FOO(in,t) value;

// RUN: c-index-test -code-completion-at=%s:4:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro}{LeftParen (}{Placeholder args}{RightParen )} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText error}{HorizontalSpace  }{Placeholder message} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText if}{HorizontalSpace  }{Placeholder condition} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText ifdef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText ifndef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number}{HorizontalSpace  }{Text "}{Placeholder filename}{Text "} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText pragma}{HorizontalSpace  }{Placeholder arguments} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText undef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: NotImplemented:{TypedText warning}{HorizontalSpace  }{Placeholder message} (40)
// RUN: c-index-test -code-completion-at=%s:5:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText define}{HorizontalSpace  }{Placeholder macro}{LeftParen (}{Placeholder args}{RightParen )} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText elif}{HorizontalSpace  }{Placeholder condition} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText else} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText endif} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText error}{HorizontalSpace  }{Placeholder message} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText if}{HorizontalSpace  }{Placeholder condition} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText ifdef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText ifndef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText import}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText include_next}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText line}{HorizontalSpace  }{Placeholder number}{HorizontalSpace  }{Text "}{Placeholder filename}{Text "} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText pragma}{HorizontalSpace  }{Placeholder arguments} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText undef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: NotImplemented:{TypedText warning}{HorizontalSpace  }{Placeholder message} (40)
// RUN: c-index-test -code-completion-at=%s:9:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: macro definition:{TypedText BAR} (40)
// CHECK-CC3: macro definition:{TypedText FOO} (40)
// RUN: c-index-test -code-completion-at=%s:11:13 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:11:14 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:11:5 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: macro definition:{TypedText BAR} (70)
// CHECK-CC4: macro definition:{TypedText FOO}{LeftParen (}{Placeholder a}{Comma , }{Placeholder b}{RightParen )} (70)
// RUN: c-index-test -code-completion-at=%s:14:5 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: NotImplemented:{TypedText const} (50)
// CHECK-CC5: NotImplemented:{TypedText double} (50)
// CHECK-CC5: NotImplemented:{TypedText enum} (50)
// CHECK-CC5: NotImplemented:{TypedText extern} (40)
// CHECK-CC5: NotImplemented:{TypedText float} (50)
// CHECK-CC5: macro definition:{TypedText FOO}{LeftParen (}{Placeholder a}{Comma , }{Placeholder b}{RightParen )} (70)
// CHECK-CC5: TypedefDecl:{TypedText id} (50)
// CHECK-CC5: NotImplemented:{TypedText inline} (40)
// CHECK-CC5: NotImplemented:{TypedText int} (50)
// CHECK-CC5: NotImplemented:{TypedText long} (50)

// Same tests as above, but with completion caching.
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:4:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:5:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:9:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:11:5 %s | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:5 %s | FileCheck -check-prefix=CHECK-CC5 %s
