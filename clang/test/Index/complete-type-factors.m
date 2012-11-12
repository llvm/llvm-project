/* Run lines are at the end, since line/column matter in this test. */

enum Color {
  Red, Green, Blue
};

enum Priority {
  Low,
  High
};

int func1(enum Color);
enum Priority func2(int);
void func3(float);
enum Priority test1(enum Priority priority, enum Color color, int integer) {
  int i = integer;
  enum Color c = color;
  return priority;
  func1(c);
  void (^block)(enum Color, int);
  block(c, 17);
  c = color;
}

@interface A
+ (void)method:(enum Color)color priority:(enum Priority)priority;
- (void)method:(enum Color)color priority:(enum Priority)priority;
@end

void test2(A *a) {
  [a method:Red priority:High];
  [A method:Red priority:Low];
}

// RUN: c-index-test -code-completion-at=%s:16:11 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (32)
// CHECK-CC1: ParmDecl:{ResultType enum Color}{TypedText color} (17)
// CHECK-CC1: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (12)
// CHECK-CC1: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (25)
// CHECK-CC1: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (32)
// CHECK-CC1: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (32)
// CHECK-CC1: VarDecl:{ResultType int}{TypedText i} (8)
// CHECK-CC1: ParmDecl:{ResultType int}{TypedText integer} (8)
// CHECK-CC1: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (32)
// CHECK-CC1: ParmDecl:{ResultType enum Priority}{TypedText priority} (17)
// CHECK-CC1: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (32)
// CHECK-CC1: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// CHECK-CC1: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (25)
// RUN: c-index-test -code-completion-at=%s:17:18 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (16)
// CHECK-CC2: VarDecl:{ResultType enum Color}{TypedText c} (8)
// CHECK-CC2: ParmDecl:{ResultType enum Color}{TypedText color} (8)
// CHECK-CC2: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (25)
// CHECK-CC2: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC2: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (16)
// CHECK-CC2: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (65)
// CHECK-CC2: VarDecl:{ResultType int}{TypedText i} (17)
// CHECK-CC2: ParmDecl:{ResultType int}{TypedText integer} (17)
// CHECK-CC2: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (65)
// CHECK-CC2: ParmDecl:{ResultType enum Priority}{TypedText priority} (34)
// CHECK-CC2: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (16)
// CHECK-CC2: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// CHECK-CC2: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (50)
// RUN: c-index-test -code-completion-at=%s:18:10 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (65)
// CHECK-CC3: VarDecl:{ResultType enum Color}{TypedText c} (34)
// CHECK-CC3: ParmDecl:{ResultType enum Color}{TypedText color} (34)
// CHECK-CC3: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (25)
// CHECK-CC3: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (12)
// CHECK-CC3: FunctionDecl:{ResultType void}{TypedText func3}{LeftParen (}{Placeholder float}{RightParen )} (50)
// CHECK-CC3: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (65)
// CHECK-CC3: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (16)
// CHECK-CC3: VarDecl:{ResultType int}{TypedText i} (17)
// CHECK-CC3: ParmDecl:{ResultType int}{TypedText integer} (17)
// CHECK-CC3: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (16)
// CHECK-CC3: ParmDecl:{ResultType enum Priority}{TypedText priority} (8)
// CHECK-CC3: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (65)
// CHECK-CC3: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// CHECK-CC3: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (12)
// RUN: c-index-test -code-completion-at=%s:19:9 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (16)
// CHECK-CC4: VarDecl:{ResultType enum Color}{TypedText c} (8)
// CHECK-CC4: ParmDecl:{ResultType enum Color}{TypedText color} (8)
// CHECK-CC4: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (25)
// CHECK-CC4: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC4: FunctionDecl:{ResultType void}{TypedText func3}{LeftParen (}{Placeholder float}{RightParen )} (50)
// CHECK-CC4: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (16)
// CHECK-CC4: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (65)
// CHECK-CC4: VarDecl:{ResultType int}{TypedText i} (17)
// CHECK-CC4: ParmDecl:{ResultType int}{TypedText integer} (17)
// CHECK-CC4: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (65)
// CHECK-CC4: ParmDecl:{ResultType enum Priority}{TypedText priority} (34)
// CHECK-CC4: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (16)
// CHECK-CC4: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// CHECK-CC4: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (50)
// RUN: c-index-test -code-completion-at=%s:21:9 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: c-index-test -code-completion-at=%s:22:7 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: VarDecl:{ResultType void (^)(enum Color, int)}{TypedText block} (34)
// CHECK-CC6: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (16)
// CHECK-CC6: VarDecl:{ResultType enum Color}{TypedText c} (8)
// CHECK-CC6: ParmDecl:{ResultType enum Color}{TypedText color} (8)
// CHECK-CC6: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (25)
// CHECK-CC6: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC6: FunctionDecl:{ResultType void}{TypedText func3}{LeftParen (}{Placeholder float}{RightParen )} (50)
// CHECK-CC6: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (16)
// CHECK-CC6: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (65)
// CHECK-CC6: VarDecl:{ResultType int}{TypedText i} (17)
// CHECK-CC6: ParmDecl:{ResultType int}{TypedText integer} (17)
// CHECK-CC6: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (65)
// CHECK-CC6: ParmDecl:{ResultType enum Priority}{TypedText priority} (34)
// CHECK-CC6: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (16)
// CHECK-CC6: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// CHECK-CC6: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (50)
// RUN: c-index-test -code-completion-at=%s:31:13 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC7 %s
// RUN: c-index-test -code-completion-at=%s:32:13 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: ParmDecl:{ResultType A *}{TypedText a} (34)
// CHECK-CC7: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (16)
// CHECK-CC7: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (25)
// CHECK-CC7: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC7: FunctionDecl:{ResultType void}{TypedText func3}{LeftParen (}{Placeholder float}{RightParen )} (50)
// CHECK-CC7: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (16)
// CHECK-CC7: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (65)
// CHECK-CC7: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (65)
// CHECK-CC7: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (16)
// CHECK-CC7: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (50)
// RUN: c-index-test -code-completion-at=%s:31:26 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC8 %s
// RUN: c-index-test -code-completion-at=%s:32:26 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: ParmDecl:{ResultType A *}{TypedText a} (34)
// CHECK-CC8: EnumConstantDecl:{ResultType enum Color}{TypedText Blue} (65)
// CHECK-CC8: FunctionDecl:{ResultType int}{TypedText func1}{LeftParen (}{Placeholder enum Color}{RightParen )} (25)
// CHECK-CC8: FunctionDecl:{ResultType enum Priority}{TypedText func2}{LeftParen (}{Placeholder int}{RightParen )} (12)
// CHECK-CC8: FunctionDecl:{ResultType void}{TypedText func3}{LeftParen (}{Placeholder float}{RightParen )} (50)
// CHECK-CC8: EnumConstantDecl:{ResultType enum Color}{TypedText Green} (65)
// CHECK-CC8: EnumConstantDecl:{ResultType enum Priority}{TypedText High} (16)
// CHECK-CC8: EnumConstantDecl:{ResultType enum Priority}{TypedText Low} (16)
// CHECK-CC8: EnumConstantDecl:{ResultType enum Color}{TypedText Red} (65)
// CHECK-CC8: FunctionDecl:{ResultType enum Priority}{TypedText test1}{LeftParen (}{Placeholder enum Priority priority}{Comma , }{Placeholder enum Color color}{Comma , }{Placeholder int integer}{RightParen )} (12)
