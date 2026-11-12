// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -fixit %t
// RUN: %clang_cc1 -x objective-c -Werror %t

void object_setClass(id, id);
Class object_getClass(id);

id rhs(void);

Class pr6302(id x123) {
  x123->isa  = 0;
  x123->isa = rhs();
  x123->isa = (id)(x123->isa);
  x123->isa = (id)x123->isa;
  x123->isa = (x123->isa);
  x123->isa = (id)(x123->isa);
  return x123->isa;
}


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
    (void)v->isa; 
    (void)w->isa;
    (void)x->isa;
    (void)y->isa;
    (void)z->isa;
    (void)u->isa;
    y->isa = 0;
    y->isa = w->isa;
    x->isa = rhs();
}
@end

