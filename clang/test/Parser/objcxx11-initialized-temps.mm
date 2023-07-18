// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics
// rdar://12788429

struct CGPoint {
  double x;
  double y;
};
typedef struct CGPoint CGPoint;

struct CGSize {
  double width;
  double height;
};
typedef struct CGSize CGSize;

struct CGRect {
  CGPoint origin;
  CGSize size;
};
typedef struct CGRect CGRect;

typedef CGRect NSRect;

void HappySetFrame(NSRect frame) {}

__attribute__((objc_root_class))
@interface NSObject @end

@implementation NSObject
- (void) sadSetFrame: (NSRect)frame {}

- (void) nothing
{
        HappySetFrame({{0,0}, {13,14}});
        [self sadSetFrame: {{0,0}, {13,14}}];
}
@end
