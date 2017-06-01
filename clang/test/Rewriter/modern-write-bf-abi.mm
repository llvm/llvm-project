// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// rdar://13138459

// -Did="void*" -DSEL="void *" -DClass="void*"
@interface NSMutableArray {
  id isa;
}
@end

typedef unsigned char BOOL;
typedef unsigned long NSUInteger;

__attribute__((visibility("hidden")))
@interface __NSArrayM : NSMutableArray {
    NSUInteger _used;
    NSUInteger _doHardRetain:1;
    NSUInteger _doWeakAccess:1;
#if __LP64__
    NSUInteger _size:62;
#else
    NSUInteger _size:30;
#endif
    NSUInteger _hasObjects:1;
    NSUInteger _hasStrongReferences:1;
#if __LP64__
    NSUInteger _offset:62;
#else
    NSUInteger _offset:30;
#endif
    unsigned long _mutations;
    id *_list;
}
@end


id __CFAllocateObject2();
BOOL objc_collectingEnabled();

@implementation __NSArrayM
+ (id)__new:(const id [])objects :(NSUInteger)count :(BOOL)hasObjects :(BOOL)hasStrong :(BOOL)transferRetain {
    __NSArrayM *newArray = (__NSArrayM *)__CFAllocateObject2();
    newArray->_size = count;
    newArray->_mutations = 1;
    newArray->_doHardRetain = (hasObjects && hasStrong);
    newArray->_doWeakAccess = (objc_collectingEnabled() && !hasStrong);
    newArray->_hasObjects = hasObjects;
    newArray->_hasStrongReferences = hasStrong;
    newArray->_list = 0;
    return *newArray->_list;
}
@end

// Test2
@interface Super {
  int ivar_super_a : 5;
}
@end

@interface A : Super {
@public
  int ivar_a : 5;
}
@end

int f0(A *a) {
  return a->ivar_a;
}

@interface A () {
@public
  int ivar_ext_a : 5;
  int ivar_ext_b : 5;
}@end

int f1(A *a) {
  return a->ivar_ext_a + a->ivar_a;
}

@interface A () {
@public
  int ivar_ext2_a : 5;
  int ivar_ext2_b : 5;
}@end

int f2(A* a) {
  return a->ivar_ext2_a + a->ivar_ext_a + a->ivar_a;
}

@implementation A {
@public
  int ivar_b : 5;
  int ivar_c : 5;
  int ivar_d : 5;
}
@end

int f3(A *a) {  
  return a->ivar_d + a->ivar_ext2_a + a->ivar_ext_a + a->ivar_a;
}

__attribute__((objc_root_class)) @interface Base
{
    struct objc_class *isa;
    int full;
    int full2: 32;
    int _refs: 8;
    int field2: 3;
    unsigned f3: 8;
    short cc;
    unsigned g: 16;
    int r2: 8;
    int r3: 8;
    int r4: 2;
    int r5: 8;
    char c;
}
@end

@implementation Base @end
