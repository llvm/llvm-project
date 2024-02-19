@interface Foo
- (int)compare:(Foo*)other;
@end

@implementation Foo
- (int)compare:(Foo*)other {
  return 0;
  (void)@encode(Foo);
}
@end

// The 'barType' referenced in the ivar declarations should be annotated as
// TypeRefs.
typedef int * barType;
@interface Bar
{
    barType iVar;
    barType iVar1, iVar2;
}
@end
@implementation Bar
- (void) method
{
    barType local = iVar;
}
@end

// The ranges for attributes are not currently stored, causing most of the
// tokens to be falsely annotated. Since there are no source ranges for
// attributes, we currently don't annotate them.
@interface IBActionTests
- (IBAction) actionMethod:(in id)arg;
- (void)foo:(int)x;
@end
extern int ibaction_test(void);
@implementation IBActionTests
- (IBAction) actionMethod:(in id)arg
{
    ibaction_test();
    [self foo:0];
}
- (void) foo:(int)x
{
  (void) x;
}
@end

// Essentially the same issue as above, but impacting code marked as IBOutlets.
@interface IBOutletTests
{
    IBOutlet char * anOutlet;
}
- (IBAction) actionMethod:(id)arg;
@property IBOutlet int * aPropOutlet;
@end

// The first 'foo:' wasn't being annotated as being part of the Objective-C
// message expression since the argument was expanded from a macro.

#define VAL 0

@interface R7974151
- (int) foo:(int)arg;
- (int) method;
@end

@implementation R7974151
- (int) foo:(int)arg {
  return arg;
}
- (int) method
{
    int local = [self foo:VAL];
    int second = [self foo:0];
    return local;
}
- (int)othermethod:(IBOutletTests *)ibt {
  return *ibt.aPropOutlet;
}
@end

@protocol Proto @end

void f() {
  (void)@protocol(Proto);
}

// Properly annotate functions and variables declared within an @implementation.
@class Rdar8595462_A;
@interface Rdar8595462_B
@end

@implementation Rdar8595462_B
Rdar8595462_A * Rdar8595462_aFunction() {
  Rdar8595462_A * localVar = 0;
  return localVar;
}
static Rdar8595462_A * Rdar8595462_staticVar;
@end

// Issues doing syntax coloring of properties
@interface Rdar8595386 {
  Foo *_foo;
}

@property (readonly, copy) Foo *foo;
@property (readonly) Foo *foo2;
@end

@implementation Rdar8595386
@synthesize foo = _foo;
@dynamic foo2;
@end

// Blocks don't get colored if annotation starts within the block itself
@interface Rdar8778404
@end

@implementation Rdar8778404
- (int)blah:(int)arg, ... { return arg; }
- (int)blarg:(int)x {
  (void)^ {
    int result = [self blah:5, x];
    Rdar8778404 *a = self;
    return 0;
  };
}
@end

@interface Rdar8062781
+ (Foo*)getB;
@property (readonly, nonatomic) Foo *blah;
@property (readonly, atomic) Foo *abah;
@end

@interface rdar9535717 {
  __weak Foo *foo;
}
@end

@interface MyClass
  @property int classProperty;
@end
@interface MyClass (abc)
  @property int categoryProperty;
@end
@interface MyClass ()
  @property int extensionProperty;
@end

typedef id<Proto> *proto_ptr;

// RUN: c-index-test -test-annotate-tokens=%s:1:1:118:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' | FileCheck %s
// CHECK: Punctuation: "@" [1:1 - 1:2] ObjCInterfaceDecl=Foo:1:12
// CHECK: Keyword: "interface" [1:2 - 1:11] ObjCInterfaceDecl=Foo:1:12
// CHECK: Identifier: "Foo" [1:12 - 1:15] ObjCInterfaceDecl=Foo:1:12
// CHECK: Punctuation: "-" [2:1 - 2:2] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Punctuation: "(" [2:3 - 2:4] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Keyword: "int" [2:4 - 2:7] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Punctuation: ")" [2:7 - 2:8] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Identifier: "compare" [2:8 - 2:15] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Punctuation: ":" [2:15 - 2:16] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Punctuation: "(" [2:16 - 2:17] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Identifier: "Foo" [2:17 - 2:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [2:20 - 2:21] ParmDecl=other:2:22 (Definition)
// CHECK: Punctuation: ")" [2:21 - 2:22] ParmDecl=other:2:22 (Definition)
// CHECK: Identifier: "other" [2:22 - 2:27] ParmDecl=other:2:22 (Definition)
// CHECK: Punctuation: ";" [2:27 - 2:28] ObjCInstanceMethodDecl=compare::2:8
// CHECK: Punctuation: "@" [3:1 - 3:2] ObjCInterfaceDecl=Foo:1:12
// CHECK: Keyword: "end" [3:2 - 3:5] ObjCInterfaceDecl=Foo:1:12
// CHECK: Punctuation: "@" [5:1 - 5:2] ObjCImplementationDecl=Foo:5:17 (Definition)
// CHECK: Keyword: "implementation" [5:2 - 5:16] ObjCImplementationDecl=Foo:5:17 (Definition)
// CHECK: Identifier: "Foo" [5:17 - 5:20] ObjCImplementationDecl=Foo:5:17 (Definition)
// CHECK: Punctuation: "-" [6:1 - 6:2] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Punctuation: "(" [6:3 - 6:4] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Keyword: "int" [6:4 - 6:7] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Punctuation: ")" [6:7 - 6:8] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Identifier: "compare" [6:8 - 6:15] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Punctuation: ":" [6:15 - 6:16] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Punctuation: "(" [6:16 - 6:17] ObjCInstanceMethodDecl=compare::6:8 (Definition)
// CHECK: Identifier: "Foo" [6:17 - 6:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [6:20 - 6:21] ParmDecl=other:6:22 (Definition)
// CHECK: Punctuation: ")" [6:21 - 6:22] ParmDecl=other:6:22 (Definition)
// CHECK: Identifier: "other" [6:22 - 6:27] ParmDecl=other:6:22 (Definition)
// CHECK: Punctuation: "{" [6:28 - 6:29] CompoundStmt=
// CHECK: Keyword: "return" [7:3 - 7:9] ReturnStmt=
// CHECK: Literal: "0" [7:10 - 7:11] IntegerLiteral=
// CHECK: Punctuation: ";" [7:11 - 7:12] CompoundStmt=
// CHECK: Punctuation: "(" [8:3 - 8:4] CStyleCastExpr=
// CHECK: Keyword: "void" [8:4 - 8:8] CStyleCastExpr=
// CHECK: Punctuation: ")" [8:8 - 8:9] CStyleCastExpr=
// CHECK: Punctuation: "@" [8:9 - 8:10] ObjCEncodeExpr=
// CHECK: Keyword: "encode" [8:10 - 8:16] ObjCEncodeExpr=
// CHECK: Punctuation: "(" [8:16 - 8:17] ObjCEncodeExpr=
// CHECK: Identifier: "Foo" [8:17 - 8:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: ")" [8:20 - 8:21] ObjCEncodeExpr=
// CHECK: Punctuation: ";" [8:21 - 8:22] CompoundStmt=
// CHECK: Punctuation: "}" [9:1 - 9:2] CompoundStmt=
// CHECK: Punctuation: "@" [10:1 - 10:2] ObjCImplementationDecl=Foo:5:17 (Definition)
// CHECK: Keyword: "end" [10:2 - 10:5]
// CHECK: Keyword: "typedef" [14:1 - 14:8]
// CHECK: Keyword: "int" [14:9 - 14:12]
// CHECK: Punctuation: "*" [14:13 - 14:14]
// CHECK: Identifier: "barType" [14:15 - 14:22] TypedefDecl=barType:14:15 (Definition)
// CHECK: Punctuation: ";" [14:22 - 14:23]
// CHECK: Punctuation: "@" [15:1 - 15:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Keyword: "interface" [15:2 - 15:11] ObjCInterfaceDecl=Bar:15:12
// CHECK: Identifier: "Bar" [15:12 - 15:15] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "{" [16:1 - 16:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Identifier: "barType" [17:5 - 17:12] TypeRef=barType:14:15
// CHECK: Identifier: "iVar" [17:13 - 17:17] ObjCIvarDecl=iVar:17:13 (Definition)
// CHECK: Punctuation: ";" [17:17 - 17:18] ObjCInterfaceDecl=Bar:15:12
// CHECK: Identifier: "barType" [18:5 - 18:12] TypeRef=barType:14:15
// CHECK: Identifier: "iVar1" [18:13 - 18:18] ObjCIvarDecl=iVar1:18:13 (Definition)
// CHECK: Punctuation: "," [18:18 - 18:19] ObjCIvarDecl=iVar2:18:20 (Definition)
// CHECK: Identifier: "iVar2" [18:20 - 18:25] ObjCIvarDecl=iVar2:18:20 (Definition)
// CHECK: Punctuation: ";" [18:25 - 18:26] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "}" [19:1 - 19:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "@" [20:1 - 20:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Keyword: "end" [20:2 - 20:5] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "@" [21:1 - 21:2] ObjCImplementationDecl=Bar:21:17 (Definition)
// CHECK: Keyword: "implementation" [21:2 - 21:16] ObjCImplementationDecl=Bar:21:17 (Definition)
// CHECK: Identifier: "Bar" [21:17 - 21:20] ObjCImplementationDecl=Bar:21:17 (Definition)
// CHECK: Punctuation: "-" [22:1 - 22:2] ObjCInstanceMethodDecl=method:22:10 (Definition)
// CHECK: Punctuation: "(" [22:3 - 22:4] ObjCInstanceMethodDecl=method:22:10 (Definition)
// CHECK: Keyword: "void" [22:4 - 22:8] ObjCInstanceMethodDecl=method:22:10 (Definition)
// CHECK: Punctuation: ")" [22:8 - 22:9] ObjCInstanceMethodDecl=method:22:10 (Definition)
// CHECK: Identifier: "method" [22:10 - 22:16] ObjCInstanceMethodDecl=method:22:10 (Definition)
// CHECK: Punctuation: "{" [23:1 - 23:2] CompoundStmt=
// CHECK: Identifier: "barType" [24:5 - 24:12] TypeRef=barType:14:15
// CHECK: Identifier: "local" [24:13 - 24:18] VarDecl=local:24:13 (Definition)
// CHECK: Punctuation: "=" [24:19 - 24:20] VarDecl=local:24:13 (Definition)
// CHECK: Identifier: "iVar" [24:21 - 24:25] MemberRefExpr=iVar:17:13
// CHECK: Punctuation: ";" [24:25 - 24:26] DeclStmt=
// CHECK: Punctuation: "}" [25:1 - 25:2] CompoundStmt=
// CHECK: Punctuation: "@" [26:1 - 26:2] ObjCImplementationDecl=Bar:21:17 (Definition)
// CHECK: Keyword: "end" [26:2 - 26:5]
// CHECK: Punctuation: "@" [31:1 - 31:2] ObjCInterfaceDecl=IBActionTests:31:12
// CHECK: Keyword: "interface" [31:2 - 31:11] ObjCInterfaceDecl=IBActionTests:31:12
// CHECK: Identifier: "IBActionTests" [31:12 - 31:25] ObjCInterfaceDecl=IBActionTests:31:12
// CHECK: Punctuation: "-" [32:1 - 32:2] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Punctuation: "(" [32:3 - 32:4] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Identifier: "IBAction" [32:4 - 32:12] macro expansion=IBAction
// CHECK: Punctuation: ")" [32:12 - 32:13] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Identifier: "actionMethod" [32:14 - 32:26] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Punctuation: ":" [32:26 - 32:27] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Punctuation: "(" [32:27 - 32:28] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Keyword: "in" [32:28 - 32:30] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Identifier: "id" [32:31 - 32:33] TypeRef=id:0:0
// CHECK: Punctuation: ")" [32:33 - 32:34] ParmDecl=arg:32:34 (Definition)
// CHECK: Identifier: "arg" [32:34 - 32:37] ParmDecl=arg:32:34 (Definition)
// CHECK: Punctuation: ";" [32:37 - 32:38] ObjCInstanceMethodDecl=actionMethod::32:1
// CHECK: Punctuation: "-" [33:1 - 33:2] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Punctuation: "(" [33:3 - 33:4] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Keyword: "void" [33:4 - 33:8] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Punctuation: ")" [33:8 - 33:9] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Identifier: "foo" [33:9 - 33:12] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Punctuation: ":" [33:12 - 33:13] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Punctuation: "(" [33:13 - 33:14] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Keyword: "int" [33:14 - 33:17] ParmDecl=x:33:18 (Definition)
// CHECK: Punctuation: ")" [33:17 - 33:18] ParmDecl=x:33:18 (Definition)
// CHECK: Identifier: "x" [33:18 - 33:19] ParmDecl=x:33:18 (Definition)
// CHECK: Punctuation: ";" [33:19 - 33:20] ObjCInstanceMethodDecl=foo::33:9
// CHECK: Punctuation: "@" [34:1 - 34:2] ObjCInterfaceDecl=IBActionTests:31:12
// CHECK: Keyword: "end" [34:2 - 34:5] ObjCInterfaceDecl=IBActionTests:31:12
// CHECK: Keyword: "extern" [35:1 - 35:7]
// CHECK: Keyword: "int" [35:8 - 35:11] FunctionDecl=ibaction_test:35:12
// CHECK: Identifier: "ibaction_test" [35:12 - 35:25] FunctionDecl=ibaction_test:35:12
// CHECK: Punctuation: "(" [35:25 - 35:26] FunctionDecl=ibaction_test:35:12
// CHECK: Keyword: "void" [35:26 - 35:30] FunctionDecl=ibaction_test:35:12
// CHECK: Punctuation: ")" [35:30 - 35:31] FunctionDecl=ibaction_test:35:12
// CHECK: Punctuation: ";" [35:31 - 35:32]
// CHECK: Punctuation: "@" [36:1 - 36:2] ObjCImplementationDecl=IBActionTests:36:17 (Definition)
// CHECK: Keyword: "implementation" [36:2 - 36:16] ObjCImplementationDecl=IBActionTests:36:17 (Definition)
// CHECK: Identifier: "IBActionTests" [36:17 - 36:30] ObjCImplementationDecl=IBActionTests:36:17 (Definition)
// CHECK: Punctuation: "-" [37:1 - 37:2] ObjCInstanceMethodDecl=actionMethod::37:14 (Definition)
// CHECK: Punctuation: "(" [37:3 - 37:4] ObjCInstanceMethodDecl=actionMethod::37:14 (Definition)
// CHECK: Identifier: "IBAction" [37:4 - 37:12] macro expansion=IBAction
// CHECK: Punctuation: ")" [37:12 - 37:13] ObjCInstanceMethodDecl=actionMethod::37:14 (Definition)
// CHECK: Identifier: "actionMethod" [37:14 - 37:26] ObjCInstanceMethodDecl=actionMethod::37:14 (Definition)
// CHECK: Punctuation: ":" [37:26 - 37:27] ObjCInstanceMethodDecl=actionMethod::37:14 (Definition)
// CHECK: Keyword: "in" [37:28 - 37:30] ObjCInstanceMethodDecl=actionMethod::37:14 (Definition)
// CHECK: Identifier: "id" [37:31 - 37:33] TypeRef=id:0:0
// CHECK: Punctuation: ")" [37:33 - 37:34] ParmDecl=arg:37:34 (Definition)
// CHECK: Identifier: "arg" [37:34 - 37:37] ParmDecl=arg:37:34 (Definition)
// CHECK: Punctuation: "{" [38:1 - 38:2] CompoundStmt=
// CHECK: Identifier: "ibaction_test" [39:5 - 39:18] DeclRefExpr=ibaction_test:35:12
// CHECK: Punctuation: "(" [39:18 - 39:19] CallExpr=ibaction_test:35:12
// CHECK: Punctuation: ")" [39:19 - 39:20] CallExpr=ibaction_test:35:12
// CHECK: Punctuation: ";" [39:20 - 39:21] CompoundStmt=
// CHECK: Punctuation: "[" [40:5 - 40:6] ObjCMessageExpr=foo::33:9
// CHECK: Identifier: "self" [40:6 - 40:10] ObjCSelfExpr=self:0:0
// CHECK: Identifier: "foo" [40:11 - 40:14] ObjCMessageExpr=foo::33:9
// CHECK: Punctuation: ":" [40:14 - 40:15] ObjCMessageExpr=foo::33:9
// CHECK: Literal: "0" [40:15 - 40:16] IntegerLiteral=
// CHECK: Punctuation: "]" [40:16 - 40:17] ObjCMessageExpr=foo::33:9
// CHECK: Punctuation: ";" [40:17 - 40:18] CompoundStmt=
// CHECK: Punctuation: "}" [41:1 - 41:2] CompoundStmt=
// CHECK: Punctuation: "-" [42:1 - 42:2] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Punctuation: "(" [42:3 - 42:4] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Keyword: "void" [42:4 - 42:8] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Punctuation: ")" [42:8 - 42:9] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Identifier: "foo" [42:10 - 42:13] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Punctuation: ":" [42:13 - 42:14] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Punctuation: "(" [42:14 - 42:15] ObjCInstanceMethodDecl=foo::42:10 (Definition)
// CHECK: Keyword: "int" [42:15 - 42:18] ParmDecl=x:42:19 (Definition)
// CHECK: Punctuation: ")" [42:18 - 42:19] ParmDecl=x:42:19 (Definition)
// CHECK: Identifier: "x" [42:19 - 42:20] ParmDecl=x:42:19 (Definition)
// CHECK: Punctuation: "{" [43:1 - 43:2] CompoundStmt=
// CHECK: Punctuation: "(" [44:3 - 44:4] CStyleCastExpr=
// CHECK: Keyword: "void" [44:4 - 44:8] CStyleCastExpr=
// CHECK: Punctuation: ")" [44:8 - 44:9] CStyleCastExpr=
// CHECK: Identifier: "x" [44:10 - 44:11] DeclRefExpr=x:42:19
// CHECK: Punctuation: ";" [44:11 - 44:12] CompoundStmt=
// CHECK: Punctuation: "}" [45:1 - 45:2] CompoundStmt=
// CHECK: Punctuation: "@" [46:1 - 46:2] ObjCImplementationDecl=IBActionTests:36:17 (Definition)
// CHECK: Keyword: "end" [46:2 - 46:5]
// CHECK: Punctuation: "@" [49:1 - 49:2] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Keyword: "interface" [49:2 - 49:11] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Identifier: "IBOutletTests" [49:12 - 49:25] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Punctuation: "{" [50:1 - 50:2] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Identifier: "IBOutlet" [51:5 - 51:13] macro expansion=IBOutlet
// CHECK: Keyword: "char" [51:14 - 51:18] ObjCIvarDecl=anOutlet:51:21 (Definition)
// CHECK: Punctuation: "*" [51:19 - 51:20] ObjCIvarDecl=anOutlet:51:21 (Definition)
// CHECK: Identifier: "anOutlet" [51:21 - 51:29] ObjCIvarDecl=anOutlet:51:21 (Definition)
// CHECK: Punctuation: ";" [51:29 - 51:30] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Punctuation: "}" [52:1 - 52:2] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Punctuation: "-" [53:1 - 53:2] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Punctuation: "(" [53:3 - 53:4] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Identifier: "IBAction" [53:4 - 53:12] macro expansion=IBAction
// CHECK: Punctuation: ")" [53:12 - 53:13] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Identifier: "actionMethod" [53:14 - 53:26] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Punctuation: ":" [53:26 - 53:27] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Punctuation: "(" [53:27 - 53:28] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Identifier: "id" [53:28 - 53:30] TypeRef=id:0:0
// CHECK: Punctuation: ")" [53:30 - 53:31] ParmDecl=arg:53:31 (Definition)
// CHECK: Identifier: "arg" [53:31 - 53:34] ParmDecl=arg:53:31 (Definition)
// CHECK: Punctuation: ";" [53:34 - 53:35] ObjCInstanceMethodDecl=actionMethod::53:1
// CHECK: Punctuation: "@" [54:1 - 54:2] ObjCPropertyDecl=aPropOutlet:54:26
// CHECK: Keyword: "property" [54:2 - 54:10] ObjCPropertyDecl=aPropOutlet:54:26
// CHECK: Identifier: "IBOutlet" [54:11 - 54:19] macro expansion=IBOutlet
// CHECK: Keyword: "int" [54:20 - 54:23] ObjCPropertyDecl=aPropOutlet:54:26
// CHECK: Punctuation: "*" [54:24 - 54:25] ObjCPropertyDecl=aPropOutlet:54:26
// CHECK: Identifier: "aPropOutlet" [54:26 - 54:37] ObjCPropertyDecl=aPropOutlet:54:26
// CHECK: Punctuation: ";" [54:37 - 54:38] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Punctuation: "@" [55:1 - 55:2] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Keyword: "end" [55:2 - 55:5] ObjCInterfaceDecl=IBOutletTests:49:12
// CHECK: Punctuation: "#" [60:1 - 60:2] preprocessing directive=
// CHECK: Identifier: "define" [60:2 - 60:8] preprocessing directive=
// CHECK: Identifier: "VAL" [60:9 - 60:12] macro definition=VAL
// CHECK: Literal: "0" [60:13 - 60:14] macro definition=VAL
// CHECK: Punctuation: "@" [62:1 - 62:2] ObjCInterfaceDecl=R7974151:62:12
// CHECK: Keyword: "interface" [62:2 - 62:11] ObjCInterfaceDecl=R7974151:62:12
// CHECK: Identifier: "R7974151" [62:12 - 62:20] ObjCInterfaceDecl=R7974151:62:12
// CHECK: Punctuation: "-" [63:1 - 63:2] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Punctuation: "(" [63:3 - 63:4] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Keyword: "int" [63:4 - 63:7] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Punctuation: ")" [63:7 - 63:8] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Identifier: "foo" [63:9 - 63:12] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Punctuation: ":" [63:12 - 63:13] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Punctuation: "(" [63:13 - 63:14] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Keyword: "int" [63:14 - 63:17] ParmDecl=arg:63:18 (Definition)
// CHECK: Punctuation: ")" [63:17 - 63:18] ParmDecl=arg:63:18 (Definition)
// CHECK: Identifier: "arg" [63:18 - 63:21] ParmDecl=arg:63:18 (Definition)
// CHECK: Punctuation: ";" [63:21 - 63:22] ObjCInstanceMethodDecl=foo::63:9
// CHECK: Punctuation: "-" [64:1 - 64:2] ObjCInstanceMethodDecl=method:64:9
// CHECK: Punctuation: "(" [64:3 - 64:4] ObjCInstanceMethodDecl=method:64:9
// CHECK: Keyword: "int" [64:4 - 64:7] ObjCInstanceMethodDecl=method:64:9
// CHECK: Punctuation: ")" [64:7 - 64:8] ObjCInstanceMethodDecl=method:64:9
// CHECK: Identifier: "method" [64:9 - 64:15] ObjCInstanceMethodDecl=method:64:9
// CHECK: Punctuation: ";" [64:15 - 64:16] ObjCInstanceMethodDecl=method:64:9
// CHECK: Punctuation: "@" [65:1 - 65:2] ObjCInterfaceDecl=R7974151:62:12
// CHECK: Keyword: "end" [65:2 - 65:5] ObjCInterfaceDecl=R7974151:62:12
// CHECK: Punctuation: "@" [67:1 - 67:2] ObjCImplementationDecl=R7974151:67:17 (Definition)
// CHECK: Keyword: "implementation" [67:2 - 67:16] ObjCImplementationDecl=R7974151:67:17 (Definition)
// CHECK: Identifier: "R7974151" [67:17 - 67:25] ObjCImplementationDecl=R7974151:67:17 (Definition)
// CHECK: Punctuation: "-" [68:1 - 68:2] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Punctuation: "(" [68:3 - 68:4] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Keyword: "int" [68:4 - 68:7] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Punctuation: ")" [68:7 - 68:8] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Identifier: "foo" [68:9 - 68:12] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Punctuation: ":" [68:12 - 68:13] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Punctuation: "(" [68:13 - 68:14] ObjCInstanceMethodDecl=foo::68:9 (Definition)
// CHECK: Keyword: "int" [68:14 - 68:17] ParmDecl=arg:68:18 (Definition)
// CHECK: Punctuation: ")" [68:17 - 68:18] ParmDecl=arg:68:18 (Definition)
// CHECK: Identifier: "arg" [68:18 - 68:21] ParmDecl=arg:68:18 (Definition)
// CHECK: Punctuation: "{" [68:22 - 68:23] CompoundStmt=
// CHECK: Keyword: "return" [69:3 - 69:9] ReturnStmt=
// CHECK: Identifier: "arg" [69:10 - 69:13] DeclRefExpr=arg:68:18
// CHECK: Punctuation: ";" [69:13 - 69:14] CompoundStmt=
// CHECK: Punctuation: "}" [70:1 - 70:2] CompoundStmt=
// CHECK: Punctuation: "-" [71:1 - 71:2] ObjCInstanceMethodDecl=method:71:9 (Definition)
// CHECK: Punctuation: "(" [71:3 - 71:4] ObjCInstanceMethodDecl=method:71:9 (Definition)
// CHECK: Keyword: "int" [71:4 - 71:7] ObjCInstanceMethodDecl=method:71:9 (Definition)
// CHECK: Punctuation: ")" [71:7 - 71:8] ObjCInstanceMethodDecl=method:71:9 (Definition)
// CHECK: Identifier: "method" [71:9 - 71:15] ObjCInstanceMethodDecl=method:71:9 (Definition)
// CHECK: Punctuation: "{" [72:1 - 72:2] CompoundStmt=
// CHECK: Keyword: "int" [73:5 - 73:8] VarDecl=local:73:9 (Definition)
// CHECK: Identifier: "local" [73:9 - 73:14] VarDecl=local:73:9 (Definition)
// CHECK: Punctuation: "=" [73:15 - 73:16] VarDecl=local:73:9 (Definition)
// CHECK: Punctuation: "[" [73:17 - 73:18] ObjCMessageExpr=foo::63:9
// CHECK: Identifier: "self" [73:18 - 73:22] ObjCSelfExpr=self:0:0
// CHECK: Identifier: "foo" [73:23 - 73:26] ObjCMessageExpr=foo::63:9
// CHECK: Punctuation: ":" [73:26 - 73:27] ObjCMessageExpr=foo::63:9
// CHECK: Identifier: "VAL" [73:27 - 73:30] macro expansion=VAL:60:9
// CHECK: Punctuation: "]" [73:30 - 73:31] ObjCMessageExpr=foo::63:9
// CHECK: Punctuation: ";" [73:31 - 73:32] DeclStmt=
// CHECK: Keyword: "int" [74:5 - 74:8] VarDecl=second:74:9 (Definition)
// CHECK: Identifier: "second" [74:9 - 74:15] VarDecl=second:74:9 (Definition)
// CHECK: Punctuation: "=" [74:16 - 74:17] VarDecl=second:74:9 (Definition)
// CHECK: Punctuation: "[" [74:18 - 74:19] ObjCMessageExpr=foo::63:9
// CHECK: Identifier: "self" [74:19 - 74:23] ObjCSelfExpr=self:0:0
// CHECK: Identifier: "foo" [74:24 - 74:27] ObjCMessageExpr=foo::63:9
// CHECK: Punctuation: ":" [74:27 - 74:28] ObjCMessageExpr=foo::63:9
// CHECK: Literal: "0" [74:28 - 74:29] IntegerLiteral=
// CHECK: Punctuation: "]" [74:29 - 74:30] ObjCMessageExpr=foo::63:9
// CHECK: Punctuation: ";" [74:30 - 74:31] DeclStmt=
// CHECK: Keyword: "return" [75:5 - 75:11] ReturnStmt=
// CHECK: Identifier: "local" [75:12 - 75:17] DeclRefExpr=local:73:9
// CHECK: Punctuation: ";" [75:17 - 75:18] CompoundStmt=
// CHECK: Punctuation: "}" [76:1 - 76:2] CompoundStmt=
// CHECK: Punctuation: "-" [77:1 - 77:2] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Punctuation: "(" [77:3 - 77:4] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Keyword: "int" [77:4 - 77:7] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Punctuation: ")" [77:7 - 77:8] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Identifier: "othermethod" [77:8 - 77:19] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Punctuation: ":" [77:19 - 77:20] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Punctuation: "(" [77:20 - 77:21] ObjCInstanceMethodDecl=othermethod::77:8 (Definition)
// CHECK: Identifier: "IBOutletTests" [77:21 - 77:34] ObjCClassRef=IBOutletTests:49:12
// CHECK: Punctuation: "*" [77:35 - 77:36] ParmDecl=ibt:77:37 (Definition)
// CHECK: Punctuation: ")" [77:36 - 77:37] ParmDecl=ibt:77:37 (Definition)
// CHECK: Identifier: "ibt" [77:37 - 77:40] ParmDecl=ibt:77:37 (Definition)
// CHECK: Punctuation: "{" [77:41 - 77:42] CompoundStmt=
// CHECK: Keyword: "return" [78:3 - 78:9] ReturnStmt=
// CHECK: Punctuation: "*" [78:10 - 78:11] UnaryOperator=
// CHECK: Identifier: "ibt" [78:11 - 78:14] DeclRefExpr=ibt:77:37
// CHECK: Punctuation: "." [78:14 - 78:15] MemberRefExpr=aPropOutlet:54:26
// CHECK: Identifier: "aPropOutlet" [78:15 - 78:26] MemberRefExpr=aPropOutlet:54:26
// CHECK: Punctuation: ";" [78:26 - 78:27] CompoundStmt=
// CHECK: Punctuation: "}" [79:1 - 79:2] CompoundStmt=
// CHECK: Punctuation: "@" [80:1 - 80:2] ObjCImplementationDecl=R7974151:67:17 (Definition)
// CHECK: Keyword: "end" [80:2 - 80:5]
// CHECK: Punctuation: "@" [82:1 - 82:2] ObjCProtocolDecl=Proto:82:11 (Definition)
// CHECK: Keyword: "protocol" [82:2 - 82:10] ObjCProtocolDecl=Proto:82:11 (Definition)
// CHECK: Identifier: "Proto" [82:11 - 82:16] ObjCProtocolDecl=Proto:82:11 (Definition)
// CHECK: Punctuation: "@" [82:17 - 82:18] ObjCProtocolDecl=Proto:82:11 (Definition)
// CHECK: Keyword: "end" [82:18 - 82:21] ObjCProtocolDecl=Proto:82:11 (Definition)
// CHECK: Keyword: "void" [84:1 - 84:5] FunctionDecl=f:84:6 (Definition)
// CHECK: Identifier: "f" [84:6 - 84:7] FunctionDecl=f:84:6 (Definition)
// CHECK: Punctuation: "(" [84:7 - 84:8] FunctionDecl=f:84:6 (Definition)
// CHECK: Punctuation: ")" [84:8 - 84:9] FunctionDecl=f:84:6 (Definition)
// CHECK: Punctuation: "{" [84:10 - 84:11] CompoundStmt=
// CHECK: Punctuation: "(" [85:3 - 85:4] CStyleCastExpr=
// CHECK: Keyword: "void" [85:4 - 85:8] CStyleCastExpr=
// CHECK: Punctuation: ")" [85:8 - 85:9] CStyleCastExpr=
// CHECK: Punctuation: "@" [85:9 - 85:10] ObjCProtocolExpr=Proto:82:1
// CHECK: Keyword: "protocol" [85:10 - 85:18] ObjCProtocolExpr=Proto:82:1
// CHECK: Punctuation: "(" [85:18 - 85:19] ObjCProtocolExpr=Proto:82:1
// CHECK: Identifier: "Proto" [85:19 - 85:24] ObjCProtocolExpr=Proto:82:1
// CHECK: Punctuation: ")" [85:24 - 85:25] ObjCProtocolExpr=Proto:82:1
// CHECK: Punctuation: ";" [85:25 - 85:26] CompoundStmt=
// CHECK: Punctuation: "}" [86:1 - 86:2] CompoundStmt=
// CHECK: Punctuation: "@" [89:1 - 89:2] ObjCInterfaceDecl=Rdar8595462_A:89:8
// CHECK: Keyword: "class" [89:2 - 89:7] ObjCInterfaceDecl=Rdar8595462_A:89:8
// CHECK: Identifier: "Rdar8595462_A" [89:8 - 89:21] ObjCClassRef=Rdar8595462_A:89:8
// CHECK: Punctuation: ";" [89:21 - 89:22]
// CHECK: Punctuation: "@" [90:1 - 90:2] ObjCInterfaceDecl=Rdar8595462_B:90:12
// CHECK: Keyword: "interface" [90:2 - 90:11] ObjCInterfaceDecl=Rdar8595462_B:90:12
// CHECK: Identifier: "Rdar8595462_B" [90:12 - 90:25] ObjCInterfaceDecl=Rdar8595462_B:90:12
// CHECK: Punctuation: "@" [91:1 - 91:2] ObjCInterfaceDecl=Rdar8595462_B:90:12
// CHECK: Keyword: "end" [91:2 - 91:5] ObjCInterfaceDecl=Rdar8595462_B:90:12
// CHECK: Punctuation: "@" [93:1 - 93:2] ObjCImplementationDecl=Rdar8595462_B:93:17 (Definition)
// CHECK: Keyword: "implementation" [93:2 - 93:16] ObjCImplementationDecl=Rdar8595462_B:93:17 (Definition)
// CHECK: Identifier: "Rdar8595462_B" [93:17 - 93:30] ObjCImplementationDecl=Rdar8595462_B:93:17 (Definition)
// CHECK: Identifier: "Rdar8595462_A" [94:1 - 94:14] ObjCClassRef=Rdar8595462_A:89:8
// CHECK: Punctuation: "*" [94:15 - 94:16] FunctionDecl=Rdar8595462_aFunction:94:17 (Definition)
// CHECK: Identifier: "Rdar8595462_aFunction" [94:17 - 94:38] FunctionDecl=Rdar8595462_aFunction:94:17 (Definition)
// CHECK: Punctuation: "(" [94:38 - 94:39] FunctionDecl=Rdar8595462_aFunction:94:17 (Definition)
// CHECK: Punctuation: ")" [94:39 - 94:40] FunctionDecl=Rdar8595462_aFunction:94:17 (Definition)
// CHECK: Punctuation: "{" [94:41 - 94:42] CompoundStmt=
// CHECK: Identifier: "Rdar8595462_A" [95:3 - 95:16] ObjCClassRef=Rdar8595462_A:89:8
// CHECK: Punctuation: "*" [95:17 - 95:18] VarDecl=localVar:95:19 (Definition)
// CHECK: Identifier: "localVar" [95:19 - 95:27] VarDecl=localVar:95:19 (Definition)
// CHECK: Punctuation: "=" [95:28 - 95:29] VarDecl=localVar:95:19 (Definition)
// CHECK: Literal: "0" [95:30 - 95:31] IntegerLiteral=
// CHECK: Punctuation: ";" [95:31 - 95:32] DeclStmt=
// CHECK: Keyword: "return" [96:3 - 96:9] ReturnStmt=
// CHECK: Identifier: "localVar" [96:10 - 96:18] DeclRefExpr=localVar:95:19
// CHECK: Punctuation: ";" [96:18 - 96:19] CompoundStmt=
// CHECK: Punctuation: "}" [97:1 - 97:2] CompoundStmt=
// CHECK: Keyword: "static" [98:1 - 98:7] VarDecl=Rdar8595462_staticVar:98:24
// CHECK: Identifier: "Rdar8595462_A" [98:8 - 98:21] ObjCClassRef=Rdar8595462_A:89:8
// CHECK: Punctuation: "*" [98:22 - 98:23] VarDecl=Rdar8595462_staticVar:98:24
// CHECK: Identifier: "Rdar8595462_staticVar" [98:24 - 98:45] VarDecl=Rdar8595462_staticVar:98:24
// CHECK: Punctuation: ";" [98:45 - 98:46] ObjCImplementationDecl=Rdar8595462_B:93:17 (Definition)
// CHECK: Punctuation: "@" [99:1 - 99:2] ObjCImplementationDecl=Rdar8595462_B:93:17 (Definition)
// CHECK: Keyword: "end" [99:2 - 99:5]

// CHECK: Punctuation: "@" [106:1 - 106:2] ObjCPropertyDecl=foo:106:33
// CHECK: Keyword: "property" [106:2 - 106:10] ObjCPropertyDecl=foo:106:33
// CHECK: Punctuation: "(" [106:11 - 106:12] ObjCPropertyDecl=foo:106:33
// CHECK: Keyword: "readonly" [106:12 - 106:20] ObjCPropertyDecl=foo:106:33
// CHECK: Punctuation: "," [106:20 - 106:21] ObjCPropertyDecl=foo:106:33
// CHECK: Keyword: "copy" [106:22 - 106:26] ObjCPropertyDecl=foo:106:33
// CHECK: Punctuation: ")" [106:26 - 106:27] ObjCPropertyDecl=foo:106:33
// CHECK: Identifier: "Foo" [106:28 - 106:31] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [106:32 - 106:33] ObjCPropertyDecl=foo:106:33
// CHECK: Identifier: "foo" [106:33 - 106:36] ObjCPropertyDecl=foo:106:33
// CHECK: Keyword: "property" [107:2 - 107:10] ObjCPropertyDecl=foo2:107:27
// CHECK: Punctuation: "(" [107:11 - 107:12] ObjCPropertyDecl=foo2:107:27
// CHECK: Keyword: "readonly" [107:12 - 107:20] ObjCPropertyDecl=foo2:107:27
// CHECK: Punctuation: ")" [107:20 - 107:21] ObjCPropertyDecl=foo2:107:27
// CHECK: Identifier: "Foo" [107:22 - 107:25] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [107:26 - 107:27] ObjCPropertyDecl=foo2:107:27
// CHECK: Identifier: "foo2" [107:27 - 107:31] ObjCPropertyDecl=foo2:107:27

// CHECK: Punctuation: "@" [111:1 - 111:2] ObjCSynthesizeDecl=foo:106:33 (Definition)
// CHECK: Keyword: "synthesize" [111:2 - 111:12] ObjCSynthesizeDecl=foo:106:33 (Definition)
// CHECK: Identifier: "foo" [111:13 - 111:16] ObjCSynthesizeDecl=foo:106:33 (Definition)
// CHECK: Punctuation: "=" [111:17 - 111:18] ObjCSynthesizeDecl=foo:106:33 (Definition)
// CHECK: Identifier: "_foo" [111:19 - 111:23] MemberRef=_foo:103:8
// CHECK: Punctuation: ";" [111:23 - 111:24] ObjCImplementationDecl=Rdar8595386:110:17 (Definition)

// RUN: c-index-test -test-annotate-tokens=%s:123:1:126:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' | FileCheck -check-prefix=CHECK-INSIDE_BLOCK %s
// CHECK-INSIDE_BLOCK: Keyword: "int" [123:5 - 123:8] VarDecl=result:123:9 (Definition)
// CHECK-INSIDE_BLOCK: Identifier: "result" [123:9 - 123:15] VarDecl=result:123:9 (Definition)
// CHECK-INSIDE_BLOCK: Punctuation: "=" [123:16 - 123:17] VarDecl=result:123:9 (Definition)
// CHECK-INSIDE_BLOCK: Punctuation: "[" [123:18 - 123:19] ObjCMessageExpr=blah::120:8
// CHECK-INSIDE_BLOCK: Identifier: "self" [123:19 - 123:23] ObjCSelfExpr=self:0:0
// CHECK-INSIDE_BLOCK: Identifier: "blah" [123:24 - 123:28] ObjCMessageExpr=blah::120:8
// CHECK-INSIDE_BLOCK: Punctuation: ":" [123:28 - 123:29] ObjCMessageExpr=blah::120:8
// CHECK-INSIDE_BLOCK: Literal: "5" [123:29 - 123:30] IntegerLiteral=
// CHECK-INSIDE_BLOCK: Punctuation: "," [123:30 - 123:31] ObjCMessageExpr=blah::120:8
// CHECK-INSIDE_BLOCK: Identifier: "x" [123:32 - 123:33] DeclRefExpr=x:121:19
// CHECK-INSIDE_BLOCK: Punctuation: "]" [123:33 - 123:34] ObjCMessageExpr=blah::120:8
// CHECK-INSIDE_BLOCK: Punctuation: ";" [123:34 - 123:35] DeclStmt=
// CHECK-INSIDE_BLOCK: Identifier: "Rdar8778404" [124:5 - 124:16] ObjCClassRef=Rdar8778404:116:12
// CHECK-INSIDE_BLOCK: Punctuation: "*" [124:17 - 124:18] VarDecl=a:124:18 (Definition)
// CHECK-INSIDE_BLOCK: Identifier: "a" [124:18 - 124:19] VarDecl=a:124:18 (Definition)
// CHECK-INSIDE_BLOCK: Punctuation: "=" [124:20 - 124:21] VarDecl=a:124:18 (Definition)
// CHECK-INSIDE_BLOCK: Identifier: "self" [124:22 - 124:26] ObjCSelfExpr=self:0:0

// RUN: c-index-test -test-annotate-tokens=%s:130:1:134:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' | FileCheck -check-prefix=CHECK-PROP-AFTER-METHOD %s
// CHECK-PROP-AFTER-METHOD: Punctuation: "@" [130:1 - 130:2] ObjCInterfaceDecl=Rdar8062781:130:12
// CHECK-PROP-AFTER-METHOD: Keyword: "interface" [130:2 - 130:11] ObjCInterfaceDecl=Rdar8062781:130:12
// CHECK-PROP-AFTER-METHOD: Identifier: "Rdar8062781" [130:12 - 130:23] ObjCInterfaceDecl=Rdar8062781:130:12
// CHECK-PROP-AFTER-METHOD: Punctuation: "+" [131:1 - 131:2] ObjCClassMethodDecl=getB:131:9
// CHECK-PROP-AFTER-METHOD: Punctuation: "(" [131:3 - 131:4] ObjCClassMethodDecl=getB:131:9
// CHECK-PROP-AFTER-METHOD: Identifier: "Foo" [131:4 - 131:7] ObjCClassRef=Foo:1:12
// CHECK-PROP-AFTER-METHOD: Punctuation: "*" [131:7 - 131:8] ObjCClassMethodDecl=getB:131:9
// CHECK-PROP-AFTER-METHOD: Punctuation: ")" [131:8 - 131:9] ObjCClassMethodDecl=getB:131:9
// CHECK-PROP-AFTER-METHOD: Identifier: "getB" [131:9 - 131:13] ObjCClassMethodDecl=getB:131:9
// CHECK-PROP-AFTER-METHOD: Punctuation: ";" [131:13 - 131:14] ObjCClassMethodDecl=getB:131:9
// CHECK-PROP-AFTER-METHOD: Punctuation: "@" [132:1 - 132:2] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Keyword: "property" [132:2 - 132:10] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Punctuation: "(" [132:11 - 132:12] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Keyword: "readonly" [132:12 - 132:20] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Punctuation: "," [132:20 - 132:21] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Keyword: "nonatomic" [132:22 - 132:31] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Punctuation: ")" [132:31 - 132:32] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Identifier: "Foo" [132:33 - 132:36] ObjCClassRef=Foo:1:12
// CHECK-PROP-AFTER-METHOD: Punctuation: "*" [132:37 - 132:38] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Identifier: "blah" [132:38 - 132:42] ObjCPropertyDecl=blah:132:38
// CHECK-PROP-AFTER-METHOD: Punctuation: ";" [132:42 - 132:43] ObjCInterfaceDecl=Rdar8062781:130:12
// CHECK-PROP-AFTER-METHOD: Punctuation: "@" [133:1 - 133:2] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Keyword: "property" [133:2 - 133:10] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Punctuation: "(" [133:11 - 133:12] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Keyword: "readonly" [133:12 - 133:20] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Punctuation: "," [133:20 - 133:21] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Keyword: "atomic" [133:22 - 133:28] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Punctuation: ")" [133:28 - 133:29] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Identifier: "Foo" [133:30 - 133:33] ObjCClassRef=Foo:1:12
// CHECK-PROP-AFTER-METHOD: Punctuation: "*" [133:34 - 133:35] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Identifier: "abah" [133:35 - 133:39] ObjCPropertyDecl=abah:133:35
// CHECK-PROP-AFTER-METHOD: Punctuation: ";" [133:39 - 133:40] ObjCInterfaceDecl=Rdar8062781:130:12
// CHECK-PROP-AFTER-METHOD: Punctuation: "@" [134:1 - 134:2] ObjCInterfaceDecl=Rdar8062781:130:12

// RUN: c-index-test -test-annotate-tokens=%s:137:1:138:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-WITH-WEAK %s
// CHECK-WITH-WEAK: Identifier: "__weak" [137:3 - 137:9] macro expansion
// CHECK-WITH-WEAK: Identifier: "Foo" [137:10 - 137:13] ObjCClassRef=Foo:1:12
// CHECK-WITH-WEAK: Punctuation: "*" [137:14 - 137:15] ObjCIvarDecl=foo:137:15 (Definition)
// CHECK-WITH-WEAK: Identifier: "foo" [137:15 - 137:18] ObjCIvarDecl=foo:137:15 (Definition)
// CHECK-WITH-WEAK: Punctuation: ";" [137:18 - 137:19] ObjCInterfaceDecl=rdar9535717:136:12
// CHECK-WITH-WEAK: Punctuation: "}" [138:1 - 138:2] ObjCInterfaceDecl=rdar9535717:136:12

// RUN: c-index-test -test-annotate-tokens=%s:141:1:149:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-PROP %s
// CHECK-PROP: Keyword: "property" [142:4 - 142:12] ObjCPropertyDecl=classProperty:142:17
// CHECK-PROP: Keyword: "int" [142:13 - 142:16] ObjCPropertyDecl=classProperty:142:17
// CHECK-PROP: Identifier: "classProperty" [142:17 - 142:30] ObjCPropertyDecl=classProperty:142:17
// CHECK-PROP: Keyword: "property" [145:4 - 145:12] ObjCPropertyDecl=categoryProperty:145:17
// CHECK-PROP: Keyword: "int" [145:13 - 145:16] ObjCPropertyDecl=categoryProperty:145:17
// CHECK-PROP: Identifier: "categoryProperty" [145:17 - 145:33] ObjCPropertyDecl=categoryProperty:145:17
// CHECK-PROP: Keyword: "property" [148:4 - 148:12] ObjCPropertyDecl=extensionProperty:148:17
// CHECK-PROP: Keyword: "int" [148:13 - 148:16] ObjCPropertyDecl=extensionProperty:148:17
// CHECK-PROP: Identifier: "extensionProperty" [148:17 - 148:34] ObjCPropertyDecl=extensionProperty:148:17

// RUN: c-index-test -test-annotate-tokens=%s:151:1:152:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-ID-PROTO %s
// CHECK-ID-PROTO: Identifier: "id" [151:9 - 151:11] TypeRef=id:0:0
// CHECK-ID-PROTO: Punctuation: "<" [151:11 - 151:12] TypedefDecl=proto_ptr:151:20 (Definition)
// CHECK-ID-PROTO: Identifier: "Proto" [151:12 - 151:17] ObjCProtocolRef=Proto
// CHECK-ID-PROTO: Punctuation: ">" [151:17 - 151:18] TypedefDecl=proto_ptr:151:20 (Definition)
