// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 

void *sel_registerName(const char *);

@interface NSMutableArray 
- (void)addObject:(id)addObject;
@end

@interface NSInvocation {
@private
    id _container;
}
+ (NSInvocation *)invocationWithMethodSignature;

@end

@implementation NSInvocation

+ (NSInvocation *)invocationWithMethodSignature {
    NSInvocation *newInv;
    id obj = newInv->_container;
    [newInv->_container addObject:0];
   return 0;
}
@end

// CHECK: id obj = (*(id *)((char *)newInv + OBJC_IVAR_$_NSInvocation$_container));
// rdar://11076938
// CHECK: struct _class_t *superclass;
// CHECK: extern "C" __declspec(dllimport) struct objc_cache _objc_empty_cache;
