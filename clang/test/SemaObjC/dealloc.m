// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -fblocks -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// rdar://11987838

@protocol NSObject
- dealloc; // expected-error {{return type must be correctly specified as 'void' under ARC, instead of 'id'}}
// CHECK: fix-it:"{{.*}}":{6:3-6:3}:"(void)"
@end

@protocol Foo <NSObject> @end

@interface Root <Foo>
@end

@interface Baz : Root {
}
@end

@implementation Baz
-  (id) dealloc { // expected-error {{return type must be correctly specified as 'void' under ARC, instead of 'id'}}
// CHECK: fix-it:"{{.*}}":{20:5-20:7}:"void"
}

@end

