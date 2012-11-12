/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */
#define IBAction void
@protocol P1
- (id)abc;
- (id)initWithInt:(int)x;
- (id)initWithTwoInts:(inout int)x second:(int)y;
- (int)getInt;
- (id)getSelf;
@end
@protocol P1;
@protocol P2<P1>
+ (id)alloc;
@end

@interface A <P1>
- (id)init;
- (int)getValue;
@end

@interface B : A<P2>
- (id)initWithInt:(int)x;
- (int)getSecondValue;
- (id)getSelf;
- (int)setValue:(int)x;
@end

@interface B (FooBar)
- (id)categoryFunction:(int)x;
@end

@implementation B
- (int)getSecondValue { return 0; }
- (id)init { return self; }
- (id)getSelf { return self; }
- (void)setValue:(int)x { }
- (id)initWithTwoInts:(int)x second:(int)y { return self; }
+ (id)alloc { return 0; }
@end

@implementation B (FooBar)
- (id)categoryFunction:(int)x { return self; }
@end

@interface C
- (int)first:(int)x second:(float)y third:(double)z;
- (id)first:(int)xx second2:(float)y2 third:(double)z;
- (void*)first:(int)xxx second3:(float)y3 third:(double)z;
@end

@interface D
- (int)first:(int)x second2:(float)y third:(double)z;
@end

@implementation D
- (int)first:(int)x second2:(float)y third:(double)z { }
@end

@interface Passing
- (oneway void)method:(in id)x;
@end

@interface Gaps
- (void)method:(int)x :(int)y;
@end

@implementation Gaps
- (void)method:(int)x :(int)y {}
@end

@implementation Passing
- (oneway void)method:(in id x) {}
@end

// RUN: c-index-test -code-completion-at=%s:17:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText abc} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText getInt} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithInt}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x} (40)
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithTwoInts}{TypedText :}{LeftParen (}{Text inout }{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second:}{LeftParen (}{Text int}{RightParen )}{Text y} (40)
// RUN: c-index-test -code-completion-at=%s:17:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInstanceMethodDecl:{TypedText abc}
// CHECK-CC2-NEXT: ObjCInstanceMethodDecl:{TypedText getSelf}
// CHECK-CC2: ObjCInstanceMethodDecl:{TypedText initWithInt}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC2: ObjCInstanceMethodDecl:{TypedText initWithTwoInts}{TypedText :}{LeftParen (}{Text inout }{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second:}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:24:7 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText abc}
// CHECK-CC3-NEXT: ObjCInstanceMethodDecl:{TypedText getSelf}
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText init}
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText initWithInt}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText initWithTwoInts}{TypedText :}{LeftParen (}{Text inout }{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second:}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:33:3 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (42)
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText getInt}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText getSecondValue}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (40)
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithInt}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithTwoInts}{TypedText :}{LeftParen (}{Text inout }{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second:}{LeftParen (}{Text int}{RightParen )}{Text y}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText setValue}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:33:8 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText getInt}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (42)
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText getSecondValue}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (40)
// CHECK-CC5-NOT: {TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText setValue}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:37:7 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (40)
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText initWithInt}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText initWithTwoInts}{TypedText :}{LeftParen (}{Text inout }{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second:}{LeftParen (}{Text int}{RightParen )}{Text y}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:42:3 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (42)
// CHECK-CC7: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText categoryFunction}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC7: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (42)
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:52:21 %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType id}{Informative first:}{TypedText second2:}{Text (float)y2}{HorizontalSpace  }{TypedText third:}{Text (double)z} (35)
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType void *}{Informative first:}{TypedText second3:}{Text (float)y3}{HorizontalSpace  }{TypedText third:}{Text (double)z} (35)
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{TypedText second:}{Text (float)y}{HorizontalSpace  }{TypedText third:}{Text (double)z} (8)
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:52:19 %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: NotImplemented:{TypedText x} (40)
// CHECK-CC9: NotImplemented:{TypedText xx} (40)
// CHECK-CC9: NotImplemented:{TypedText xxx} (40)
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:52:36 %s | FileCheck -check-prefix=CHECK-CCA %s
// CHECK-CCA: NotImplemented:{TypedText y2} (40)
// RUN: c-index-test -code-completion-at=%s:56:3 %s | FileCheck -check-prefix=CHECK-CCB %s
// CHECK-CCB: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText first}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second2:}{LeftParen (}{Text float}{RightParen )}{Text y}{HorizontalSpace  }{TypedText third:}{LeftParen (}{Text double}{RightParen )}{Text z} (40)
// RUN: c-index-test -code-completion-at=%s:56:8 %s | FileCheck -check-prefix=CHECK-CCC %s
// CHECK-CCC: ObjCInstanceMethodDecl:{TypedText first}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText second2:}{LeftParen (}{Text float}{RightParen )}{Text y}{HorizontalSpace  }{TypedText third:}{LeftParen (}{Text double}{RightParen )}{Text z} (40)
// RUN: c-index-test -code-completion-at=%s:56:21 %s | FileCheck -check-prefix=CHECK-CCD %s
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType id}{Informative first:}{TypedText second2:}{Text (float)y2}{HorizontalSpace  }{TypedText third:}{Text (double)z} (35)
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{TypedText second2:}{Text (float)y}{HorizontalSpace  }{TypedText third:}{Text (double)z} (8)
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType void *}{Informative first:}{TypedText second3:}{Text (float)y3}{HorizontalSpace  }{TypedText third:}{Text (double)z} (35)
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{TypedText second:}{Text (float)y}{HorizontalSpace  }{TypedText third:}{Text (double)z} (8)
// RUN: c-index-test -code-completion-at=%s:56:38 %s | FileCheck -check-prefix=CHECK-CCE %s
// CHECK-CCE: ObjCInstanceMethodDecl:{ResultType id}{Informative first:}{Informative second2:}{TypedText third:}{Text (double)z} (35)
// CHECK-CCE: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{Informative second2:}{TypedText third:}{Text (double)z} (8)
// RUN: c-index-test -code-completion-at=%s:60:4 %s | FileCheck -check-prefix=CHECK-CCF %s
// CHECK-CCF: ObjCInterfaceDecl:{TypedText A} (50)
// CHECK-CCF: ObjCInterfaceDecl:{TypedText B} (50)
// CHECK-CCF: NotImplemented:{TypedText bycopy} (40)
// CHECK-CCF: NotImplemented:{TypedText byref} (40)
// CHECK-CCF: NotImplemented:{TypedText in} (40)
// CHECK-CCF: NotImplemented:{TypedText inout} (40)
// CHECK-CCF: NotImplemented:{TypedText oneway} (40)
// CHECK-CCF: NotImplemented:{TypedText out} (40)
// CHECK-CCF: NotImplemented:{TypedText unsigned} (50)
// CHECK-CCF: NotImplemented:{TypedText void} (50)
// CHECK-CCF: NotImplemented:{TypedText volatile} (50)
// RUN: c-index-test -code-completion-at=%s:60:11 %s | FileCheck -check-prefix=CHECK-CCG %s
// CHECK-CCG: ObjCInterfaceDecl:{TypedText A} (50)
// CHECK-CCG: ObjCInterfaceDecl:{TypedText B} (50)
// CHECK-CCG-NOT: NotImplemented:{TypedText bycopy} (40)
// CHECK-CCG-NOT: NotImplemented:{TypedText byref} (40)
// CHECK-CCG: NotImplemented:{TypedText in} (40)
// CHECK-CCG: NotImplemented:{TypedText inout} (40)
// CHECK-CCG-NOT: NotImplemented:{TypedText oneway} (40)
// CHECK-CCG: NotImplemented:{TypedText out} (40)
// CHECK-CCG: NotImplemented:{TypedText unsigned} (50)
// CHECK-CCG: NotImplemented:{TypedText void} (50)
// CHECK-CCG: NotImplemented:{TypedText volatile} (50)
// RUN: c-index-test -code-completion-at=%s:60:24 %s | FileCheck -check-prefix=CHECK-CCF %s
// RUN: c-index-test -code-completion-at=%s:60:26 %s | FileCheck -check-prefix=CHECK-CCH %s
// CHECK-CCH: ObjCInterfaceDecl:{TypedText A} (50)
// CHECK-CCH: ObjCInterfaceDecl:{TypedText B} (50)
// CHECK-CCH: NotImplemented:{TypedText bycopy} (40)
// CHECK-CCH: NotImplemented:{TypedText byref} (40)
// CHECK-CCH-NOT: NotImplemented:{TypedText in} (40)
// CHECK-CCH: NotImplemented:{TypedText inout} (40)
// CHECK-CCH: NotImplemented:{TypedText oneway} (40)
// CHECK-CCH: NotImplemented:{TypedText out} (40)
// CHECK-CCH: NotImplemented:{TypedText unsigned} (50)
// CHECK-CCH: NotImplemented:{TypedText void} (50)
// CHECK-CCH: NotImplemented:{TypedText volatile} (50)

// IBAction completion
// RUN: c-index-test -code-completion-at=%s:5:4 %s | FileCheck -check-prefix=CHECK-IBACTION %s
// CHECK-IBACTION: NotImplemented:{TypedText IBAction}{RightParen )}{Placeholder selector}{Colon :}{LeftParen (}{Text id}{RightParen )}{Text sender} (40)

// <rdar://problem/8939352>
// RUN: c-index-test -code-completion-at=%s:68:9 %s | FileCheck -check-prefix=CHECK-8939352 %s
// CHECK-8939352: ObjCInstanceMethodDecl:{TypedText method}{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{TypedText :}{LeftParen (}{Text int}{RightParen )}{Text y} (40)


// RUN: c-index-test -code-completion-at=%s:72:2 %s | FileCheck -check-prefix=CHECK-ONEWAY %s
// CHECK-ONEWAY: ObjCInstanceMethodDecl:{LeftParen (}{Text oneway }{Text void}{RightParen )}{TypedText method}{TypedText :}{LeftParen (}{Text in }{Text id}{RightParen )}{Text x} (40)
