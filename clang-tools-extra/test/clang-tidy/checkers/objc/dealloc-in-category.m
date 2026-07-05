// RUN: %check_clang_tidy %s objc-dealloc-in-category %t

@interface NSObject
// Used to quash warning about missing base class.
- (void)dealloc;
@end

@interface Foo : NSObject
@end

@implementation Foo
- (void)dealloc {
  // No warning should be generated here.
}
@end

@interface Bar : NSObject
@end

@interface Bar (BarCategory)
@end

@implementation Bar (BarCategory)
+ (void)dealloc {
  // Should not trigger on class methods.
}

- (void)dealloc {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: category 'BarCategory' should not implement -dealloc [objc-dealloc-in-category]
}
@end

@interface Baz : NSObject
@end

@implementation Baz
- (void)dealloc {
  // Should not trigger on implementation in the class itself, even with
  // it declared in the category (below).
}
@end

@interface Baz (BazCategory)
// A declaration in a category @interface does not by itself provide an
// overriding implementation, and should not generate a warning.
- (void)dealloc;
@end
