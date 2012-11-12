typedef signed char BOOL;
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#define bool _Bool
@interface NSArray + (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt; @end
@interface A
- (int)method:(id)param1;
@property int prop1;
@end
__strong id global;
@implementation A
- (int)method:(id)param1 {
  void foo(bool (^block)(id x, A* y));
  for(BOOL B = YES; ; ) { }
  @[ global ];
  @{ global : global };
}
@end

// RUN: c-index-test -code-completion-at=%s:13:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{ResultType NSString *}{TypedText @"}{Placeholder string}{Text "} (40)
// CHECK-CC1: NotImplemented:{ResultType id}{TypedText @(}{Placeholder expression}{RightParen )} (40)
// CHECK-CC1: NotImplemented:{ResultType NSArray *}{TypedText @[}{Placeholder objects, ...}{RightBracket ]} (40)
// CHECK-CC1: NotImplemented:{ResultType NSDictionary *}{TypedText @{}{Placeholder key}{Colon :}{HorizontalSpace  }{Placeholder object, ...}{RightBrace }} (40)
// CHECK-CC1: NotImplemented:{ResultType SEL}{TypedText _cmd} (80)
// CHECK-CC1: TypedefDecl:{TypedText BOOL} (50)
// CHECK-CC1: macro definition:{TypedText bool} (51)
// CHECK-CC1: macro definition:{TypedText NO} (65)
// CHECK-CC1: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC1: macro definition:{TypedText YES} (65)
// RUN: c-index-test -code-completion-at=%s:14:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: TypedefDecl:{TypedText BOOL} (50)
// CHECK-CC2: NotImplemented:{TypedText char} (50)
// CHECK-CC2: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:15:1 -fobjc-arc -fobjc-nonfragile-abi  %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:15:1 -fobjc-arc -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: FunctionDecl:{ResultType void}{TypedText foo}{LeftParen (}{Placeholder ^bool(id x, A *y)block}{RightParen )} (34)
// CHECK-CC3: VarDecl:{ResultType id}{TypedText global} (50)
// CHECK-CC3: ParmDecl:{ResultType id}{TypedText param1} (34)

// RUN: c-index-test -code-completion-at=%s:15:5 %s | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: c-index-test -code-completion-at=%s:16:5 %s | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: c-index-test -code-completion-at=%s:16:14 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: NotImplemented:{ResultType NSArray *}{TypedText @[}{Placeholder objects, ...}{RightBracket ]} (40)
// CHECK-CC4: NotImplemented:{ResultType NSDictionary *}{TypedText @{}{Placeholder key}{Colon :}{HorizontalSpace  }{Placeholder object, ...}{RightBrace }} (40)
// CHECK-CC4: NotImplemented:{ResultType SEL}{TypedText _cmd} (80)
// CHECK-CC4: macro definition:{TypedText bool} (51)
// CHECK-CC4: macro definition:{TypedText NO} (65)
// CHECK-CC4: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC4: macro definition:{TypedText YES} (65)
