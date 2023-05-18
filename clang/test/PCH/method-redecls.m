// RUN: %clang_cc1 -x objective-c %s -emit-pch -o %t
// RUN: %clang_cc1 -x objective-c %s -emit-pch -o %t -D IMPL

// Avoid infinite loop because of method redeclarations.

@interface Foo
-(void)meth;
-(void)meth;
-(void)meth;
@end

#ifdef IMPL

@implementation Foo
-(void)meth { }
@end

#endif
