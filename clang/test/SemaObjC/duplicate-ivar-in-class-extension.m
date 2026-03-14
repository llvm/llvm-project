// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Root @end

@interface SuperClass  : Root 
{
  int iSuper;	// expected-note {{previous declaration is here}}
}
@end

@interface SubClass : SuperClass {
    int ivar;	// expected-error {{duplicate member 'ivar'}}
    int another_ivar;	// expected-error {{duplicate member 'another_ivar'}}
    int iSuper;	// expected-error {{duplicate member 'iSuper'}}
}
@end

@interface SuperClass () {
   int ivar;	// expected-note {{previous declaration is here}}
}
@end

@interface Root () {
  int another_ivar;	// expected-note {{previous declaration is here}}
}
@end

@implementation SubClass
-(int) method {
        return self->ivar;  // would be ambiguous if the duplicate ivar were allowed
}
@end
