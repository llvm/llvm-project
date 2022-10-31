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
// CHECK-CC1: Pattern:{TypedText define}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText define}{HorizontalSpace  }{Placeholder macro}{LeftParen (}{Placeholder args}{RightParen )} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText error}{HorizontalSpace  }{Placeholder message} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText if}{HorizontalSpace  }{Placeholder condition} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText ifdef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText ifndef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText import}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText import}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText include}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText include}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText include_next}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText include_next}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText line}{HorizontalSpace  }{Placeholder number} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText line}{HorizontalSpace  }{Placeholder number}{HorizontalSpace  }{Text "}{Placeholder filename}{Text "} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText pragma}{HorizontalSpace  }{Placeholder arguments} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText undef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC1-NEXT: Pattern:{TypedText warning}{HorizontalSpace  }{Placeholder message} (40)
// RUN: c-index-test -code-completion-at=%s:5:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: Pattern:{TypedText define}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText define}{HorizontalSpace  }{Placeholder macro}{LeftParen (}{Placeholder args}{RightParen )} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText elif}{HorizontalSpace  }{Placeholder condition} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText elifdef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText elifndef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText else} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText endif} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText error}{HorizontalSpace  }{Placeholder message} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText if}{HorizontalSpace  }{Placeholder condition} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText ifdef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText ifndef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText import}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText import}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText include}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText include}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText include_next}{HorizontalSpace  }{Text "}{Placeholder header}{Text "} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText include_next}{HorizontalSpace  }{Text <}{Placeholder header}{Text >} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText line}{HorizontalSpace  }{Placeholder number} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText line}{HorizontalSpace  }{Placeholder number}{HorizontalSpace  }{Text "}{Placeholder filename}{Text "} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText pragma}{HorizontalSpace  }{Placeholder arguments} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText undef}{HorizontalSpace  }{Placeholder macro} (40)
// CHECK-CC2-NEXT: Pattern:{TypedText warning}{HorizontalSpace  }{Placeholder message} (40)
// RUN: c-index-test -code-completion-at=%s:9:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: macro definition:{TypedText BAR} (40)
// CHECK-CC3: macro definition:{TypedText FOO} (40)
// RUN: c-index-test -code-completion-at=%s:11:13 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:11:14 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:11:5 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: macro definition:{TypedText BAR} (70)
// CHECK-CC4: macro definition:{TypedText FOO}{LeftParen (}{Placeholder a}{Comma , }{Placeholder b}{RightParen )} (70)
// RUN: c-index-test -code-completion-at=%s:14:5 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: Keyword:{TypedText const} (50)
// CHECK-CC5: Keyword:{TypedText double} (50)
// CHECK-CC5: Keyword:{TypedText enum} (50)
// CHECK-CC5: Keyword:{TypedText extern} (40)
// CHECK-CC5: Keyword:{TypedText float} (50)
// CHECK-CC5: macro definition:{TypedText FOO}{LeftParen (}{Placeholder a}{Comma , }{Placeholder b}{RightParen )} (70)
// CHECK-CC5: TypedefDecl:{TypedText id} (50)
// CHECK-CC5: Keyword:{TypedText inline} (40)
// CHECK-CC5: Keyword:{TypedText int} (50)
// CHECK-CC5: Keyword:{TypedText long} (50)

// Same tests as above, but with completion caching.
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:4:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:5:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:9:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:11:5 %s | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:5 %s | FileCheck -check-prefix=CHECK-CC5 %s
