// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://8155806

@interface Subclass 
{
    int setterOnly;
}
- (void) setSetterOnly : (int) arg;
@end

int func (int arg, Subclass *x) {
    if (x.setterOnly) { // expected-error {{expected getter method not found on object of type 'Subclass *'}}
      x.setterOnly = 1;
    }
    func(x.setterOnly + 1, x); // expected-error {{expected getter method not found on object of type 'Subclass *'}} 
    int i = x.setterOnly + 1;  // expected-error {{expected getter method not found on object of type 'Subclass *'}} 
    return x.setterOnly + 1;   // expected-error {{expected getter method not found on object of type 'Subclass *'}} 
}

