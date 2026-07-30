// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -Wdealloc-in-category -verify %s
// RUN: not %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -Wdealloc-in-category -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

@protocol NSObject
- dealloc; // expected-error {{return type must be correctly specified as 'void' under ARC, instead of 'id'}}
// CHECK: fix-it:"{{.*}}":{5:3-5:3}:"(void)"
@end

@protocol Foo <NSObject> @end

@interface Root <Foo>
@end

@interface Baz : Root {
}
@end

@implementation Baz
-  (id) dealloc { // expected-error {{return type must be correctly specified as 'void' under ARC, instead of 'id'}}
// CHECK: fix-it:"{{.*}}":{19:5-19:7}:"void"
}

@end

@interface Base
- (void)dealloc;
@end

@interface Subclass : Base
@end 

@interface Subclass (CAT)
- (void)dealloc;
@end

@implementation Subclass (CAT)
- (void)dealloc { // expected-warning {{-dealloc is being overridden in a category}}
}
@end

@interface NSObject @end
@interface NSError:NSObject
@end

@interface NSError(CAT)
- (NSError *)MCCopyAsPrimaryError __attribute__((objc_method_family(new)));
@end
@implementation NSError(CAT)
- (NSError *)MCCopyAsPrimaryError {
  return 0;
}
@end
