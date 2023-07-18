// RUN: %clang_analyze_cc1 -verify -Wno-objc-root-class %s \
// RUN:   -Wno-tautological-pointer-compare \
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

NestedNonnullMember *foo(void);

void f1(NestedNonnullMember *Root) {
  NestedNonnullMember *Grandson = Root->Child->Child;

  clang_analyzer_eval(Root->Value != 0);         // expected-warning{{TRUE}}
  clang_analyzer_eval(Grandson->Value != 0);     // expected-warning{{TRUE}}
  clang_analyzer_eval(foo()->Child->Value != 0); // expected-warning{{TRUE}}
}

// Check that we correctly process situations when non-pointer parameters
// get nonnul attributes.
// Original problem: rdar://problem/63150074
typedef struct {
  long a;
} B;
__attribute__((nonnull)) void c(B x, int *y);

void c(B x, int *y) {
  clang_analyzer_eval(y != 0); // expected-warning{{TRUE}}
}
