/* Run lines are at the end, since line/column matter in this test. */
@interface MyClass { @public }
@end

@implementation MyClass
@end

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:2:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText class}{HorizontalSpace  }{Placeholder name}
// CHECK-CC1: {TypedText compatibility_alias}{HorizontalSpace  }{Placeholder alias}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText implementation}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText interface}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText protocol}{HorizontalSpace  }{Placeholder protocol}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:3:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText end}
// CHECK-CC2: {TypedText optional}
// CHECK-CC2: {TypedText property}
// CHECK-CC2: {TypedText required}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:6:2 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: {TypedText dynamic}{HorizontalSpace  }{Placeholder property}
// CHECK-CC3: {TypedText end}
// CHECK-CC3: {TypedText synthesize}{HorizontalSpace  }{Placeholder property}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:2:1 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: Pattern:{TypedText @class}{HorizontalSpace  }{Placeholder name}
// CHECK-CC4: Pattern:{TypedText @compatibility_alias}{HorizontalSpace  }{Placeholder alias}{HorizontalSpace  }{Placeholder class}
// CHECK-CC4: Pattern:{TypedText @implementation}{HorizontalSpace  }{Placeholder class}
// CHECK-CC4: Pattern:{TypedText @interface}{HorizontalSpace  }{Placeholder class}
// CHECK-CC4: Pattern:{TypedText @protocol}{HorizontalSpace  }{Placeholder protocol}
// CHECK-CC4: Keyword:{TypedText _Bool}
// CHECK-CC4: TypedefDecl:{TypedText Class}
// CHECK-CC4: TypedefDecl:{TypedText id}
// CHECK-CC4: TypedefDecl:{TypedText SEL}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:3:1 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: {TypedText @end}
// CHECK-CC5: {TypedText @optional}
// CHECK-CC5: {TypedText @property}
// CHECK-CC5: {TypedText @required}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:2:23 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: Keyword:{TypedText package}
// CHECK-CC6: Keyword:{TypedText private}
// CHECK-CC6: Keyword:{TypedText protected}
// CHECK-CC6: Keyword:{TypedText public}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:2:22 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: Keyword:{TypedText @package}
// CHECK-CC7: Keyword:{TypedText @private}
// CHECK-CC7: Keyword:{TypedText @protected}
// CHECK-CC7: Keyword:{TypedText @public}
// CHECK-CC7: Keyword:{TypedText _Bool}
