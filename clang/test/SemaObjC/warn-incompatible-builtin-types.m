// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://7634850

@interface Foo
- (void)foo:(Class)class; // expected-note{{passing argument to parameter 'class' here}}
@end

void FUNC() {
    Class c, c1;
    SEL s1, s2;
    id i, i1;
    Foo *f;
    [f foo:f];	// expected-warning {{incompatible pointer types sending 'Foo *' to parameter of type 'Class'}}
    c = f;	// expected-warning {{incompatible pointer types assigning to 'Class' from 'Foo *'}}

    c = i;

    i = c;

    c = c1;

    i = i1;

    s1 = i;	// expected-warning {{incompatible pointer types assigning to 'SEL' from 'id'}}
    i = s1;	// expected-warning {{incompatible pointer types assigning to 'id' from 'SEL'}}

    s1 = s2;

    s1 = c;	// expected-warning {{incompatible pointer types assigning to 'SEL' from 'Class'}}

    c = s1;	// expected-warning {{incompatible pointer types assigning to 'Class' from 'SEL'}}

    f = i;

    f = c;	// expected-warning {{incompatible pointer types assigning to 'Foo *' from 'Class'}}

    f = s1;	// expected-warning {{incompatible pointer types assigning to 'Foo *' from 'SEL'}}

    i = f;

    s1 = f; 	// expected-warning {{incompatible pointer types assigning to 'SEL' from 'Foo *'}}
}
