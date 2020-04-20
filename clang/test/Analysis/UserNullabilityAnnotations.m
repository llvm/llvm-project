// RUN: %clang_analyze_cc1 -verify -Wno-objc-root-class %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=nullability \
// RUN:   -analyzer-checker=debug.ExprInspection

void clang_analyzer_eval(int);

@interface TestFunctionLevelAnnotations
- (void)method1:(int *_Nonnull)x;
- (void)method2:(int *)x __attribute__((nonnull));
@end

@implementation TestFunctionLevelAnnotations
- (void)method1:(int *_Nonnull)x {
  clang_analyzer_eval(x != 0); // expected-warning{{TRUE}}
}

- (void)method2:(int *)x {
  clang_analyzer_eval(x != 0); // expected-warning{{TRUE}}
}
@end

typedef struct NestedNonnullMember {
  struct NestedNonnullMember *Child;
  int *_Nonnull Value;
} NestedNonnullMember;

NestedNonnullMember *foo();

void f1(NestedNonnullMember *Root) {
  NestedNonnullMember *Grandson = Root->Child->Child;

  clang_analyzer_eval(Root->Value != 0);         // expected-warning{{TRUE}}
  clang_analyzer_eval(Grandson->Value != 0);     // expected-warning{{TRUE}}
  clang_analyzer_eval(foo()->Child->Value != 0); // expected-warning{{TRUE}}
}
