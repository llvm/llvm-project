// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s

//====------------------------------------------------------------====//
// Test deprecated direct usage of the 'isa' pointer.
//====------------------------------------------------------------====//

typedef unsigned long NSUInteger;

typedef struct objc_object {
  struct objc_class *isa;
} *id;

@interface NSObject {
  id firstobj;
  struct objc_class *isa;
}
- (id)performSelector:(SEL)aSelector;;
@end
@interface Whatever : NSObject
+self;
-(id)foo;
@end

static void func(void) {
 
  id x;

  // rdar://8290002
  [(*x).isa self]; // expected-warning {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
  [x->isa self]; // expected-warning {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
  
  Whatever *y;

  // GCC allows this, with the following warning: 
  //   instance variable 'isa' is @protected; this will be a hard error in the future
  //
  // FIXME: see if we can avoid the warning that follows the error.
  [(*y).isa self]; // expected-error {{instance variable 'isa' is protected}} \
                      expected-warning{{receiver type 'struct objc_class *' is not 'id' or interface pointer, consider casting it to 'id'}}
  [y->isa self]; // expected-error {{instance variable 'isa' is protected}} \
                    expected-warning{{receiver type 'struct objc_class *' is not 'id' or interface pointer, consider casting it to 'id'}}
}

// rdar://11702488
// If an ivar is (1) the first ivar in a root class and (2) named `isa`,
// then it should get the same warnings that id->isa gets.

@interface BaseClass {
@public
    Class isa; // expected-note 4 {{instance variable is declared here}}
}
@end

@interface OtherClass {
@public
    id    firstIvar;
    Class isa; // note, not first ivar;
}
@end

@interface Subclass : BaseClass @end

@interface SiblingClass : BaseClass @end

@interface Root @end

@interface hasIsa : Root {
@public
  Class isa; // note, isa is not in root class
}
@end

@implementation Subclass
-(void)method {
    hasIsa *u;
    id v;
    BaseClass *w;
    Subclass *x;
    SiblingClass *y;
    OtherClass *z;
    (void)v->isa; // expected-warning {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
    (void)w->isa; // expected-warning {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
    (void)x->isa; // expected-warning {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
    (void)y->isa; // expected-warning {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
    (void)z->isa;
    (void)u->isa;

    w->isa = 0; // expected-warning {{assignment to Objective-C's isa is deprecated in favor of object_setClass()}}
}
@end

// Test for introspection of Objective-C pointers via bitmasking.

void testBitmasking(NSObject *p) {
  (void) (((NSUInteger) p) & 0x1); // expected-warning {{bitmasking for introspection of Objective-C object pointers is strongly discouraged}}
  (void) (0x1 & ((NSUInteger) p)); // expected-warning {{bitmasking for introspection of Objective-C object pointers is strongly discouraged}}
  (void) (((NSUInteger) p) ^ 0x1); // no-warning
  (void) (0x1 ^ ((NSUInteger) p)); // no-warning
  (void) (0x1 & ((NSUInteger) [p performSelector:@selector(foo)])); // expected-warning {{bitmasking for introspection of Objective-C object pointers is strongly discouraged}}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-objc-pointer-introspection-performSelector"
  (void) (0x1 & ((NSUInteger) [p performSelector:@selector(foo)])); // no-warning
#pragma clang diagnostic pop
}