// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef struct objc_object {
  struct objc_class *isa;
} *id;

@interface NSObject {
  id firstobj;
  struct objc_class *isa;
}
@end
@interface Whatever : NSObject
+self;
@end

static void func() {
 
  id x;

  // rdar://8290002
  [(*x).isa self]; // expected-warning {{direct access to objective-c's isa is deprecated in favor of object_setClass() and object_getClass()}}
  [x->isa self]; // expected-warning {{direct access to objective-c's isa is deprecated in favor of object_setClass() and object_getClass()}}
  
  Whatever *y;

  // GCC allows this, with the following warning: 
  //   instance variable 'isa' is @protected; this will be a hard error in the future
  //
  // FIXME: see if we can avoid the 2 warnings that follow the error.
  [(*y).isa self]; // expected-error {{instance variable 'isa' is protected}} \
                      expected-warning{{receiver type 'struct objc_class *' is not 'id' or interface pointer, consider casting it to 'id'}} \
                      expected-warning{{method '-self' not found (return type defaults to 'id')}}
  [y->isa self]; // expected-error {{instance variable 'isa' is protected}} \
                    expected-warning{{receiver type 'struct objc_class *' is not 'id' or interface pointer, consider casting it to 'id'}} \
                    expected-warning{{method '-self' not found (return type defaults to 'id')}}
}

// rdar://11702488
// If an ivar is (1) the first ivar in a root class and (2) named `isa`,
// then it should get the same warnings that id->isa gets.

@interface BaseClass {
@public
    Class isa; // expected-note 3 {{instance variable is declared here}}
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
    (void)v->isa; // expected-warning {{direct access to objective-c's isa is deprecated}}
    (void)w->isa; // expected-warning {{direct access to objective-c's isa is deprecated}}
    (void)x->isa; // expected-warning {{direct access to objective-c's isa is deprecated}}
    (void)y->isa; // expected-warning {{direct access to objective-c's isa is deprecated}}
    (void)z->isa;
    (void)u->isa;
}
@end

