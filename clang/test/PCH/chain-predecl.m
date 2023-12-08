// RUN: %clang_cc1 -emit-pch -o %t1 %S/chain-predecl.h -x objective-c
// RUN: %clang_cc1 -emit-pch -o %t2 %s -x objective-c -include-pch %t1

// Test predeclarations across chained PCH.
@interface Foo
-(void)bar;
@end
@interface Boom
-(void)bar;
@end
@protocol Pro
-(void)baz;
@end
@protocol Kaboom
-(void)baz;
@end
