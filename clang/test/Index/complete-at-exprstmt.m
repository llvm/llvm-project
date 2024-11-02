/* The run lines are below, because this test is line- and
   column-number sensitive. */
@interface MyClass { int ivar; }
- (int)myMethod:(int)arg;
@end

@implementation MyClass 
- (int)myMethod:(int)arg {
  @synchronized (@encode(MyClass)) { }
}
@end

@interface A
+ (int)add:(int)x to:(int)y;
+ (int)add:(int)x to:(int)y plus:(int)z;
@end

void f() {
  @selector(add:to:);
}

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:9:4 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC1: {TypedText protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC1: {TypedText selector}{LeftParen (}{Placeholder selector}{RightParen )}
// CHECK-CC1: {TypedText synchronized}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC1: {TypedText throw}{HorizontalSpace  }{Placeholder expression}
// CHECK-CC1: {TypedText try}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @catch}{LeftParen (}{Placeholder parameter}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @finally}{LeftBrace {}{Placeholder statements}{RightBrace }}
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:9:19 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC2: {TypedText protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC2: {TypedText selector}{LeftParen (}{Placeholder selector}{RightParen )}
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:9:3 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: Pattern:{ResultType char[]}{TypedText @encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC3: Pattern:{ResultType Protocol *}{TypedText @protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC3: Pattern:{ResultType SEL}{TypedText @selector}{LeftParen (}{Placeholder selector}{RightParen )}
// CHECK-CC3: Pattern:{TypedText @synchronized}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC3: Pattern:{TypedText @throw}{HorizontalSpace  }{Placeholder expression}
// CHECK-CC3: Pattern:{TypedText @try}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @catch}{LeftParen (}{Placeholder parameter}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @finally}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC3: Declaration:{ResultType SEL}{TypedText _cmd}
// CHECK-CC3: ParmDecl:{ResultType int}{TypedText arg}
// CHECK-CC3: TypedefDecl:{TypedText Class}
// CHECK-CC3: TypedefDecl:{TypedText id}
// CHECK-CC3: ObjCIvarDecl:{ResultType int}{TypedText ivar}
// CHECK-CC3: ObjCInterfaceDecl:{TypedText MyClass}
// CHECK-CC3: TypedefDecl:{TypedText SEL}
// CHECK-CC3: Declaration:{ResultType MyClass *}{TypedText self}
// RUN: c-index-test -code-completion-at=%s:19:13 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: Pattern:{TypedText add:to:} (40)
// CHECK-CC4: Pattern:{TypedText add:to:plus:} (40)
// CHECK-CC4: Pattern:{TypedText myMethod:} (40)
// RUN: c-index-test -code-completion-at=%s:19:17 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: Pattern:{Informative add:}{TypedText to:} (40)
// CHECK-CC5: Pattern:{Informative add:}{TypedText to:plus:} (40)

