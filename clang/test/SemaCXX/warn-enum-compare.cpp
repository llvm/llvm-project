// RUN: %clang_cc1 %s -fsyntax-only -verify -triple %itanium_abi_triple
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple %ms_abi_triple -DMSABI

enum Foo { FooA, FooB, FooC };
enum Bar { BarD, BarE, BarF };
enum { AnonAA = 42, AnonAB = 43 };
enum { AnonBA = 44, AnonBB = 45 };
enum { Anon1, Anon2, Anon3 };
typedef enum { TD1, TD2 } TD;

namespace name1 {
  enum Foo {F1, F2, F3};
  enum Baz {B1, B2, B3};
}

namespace name2 {
  enum Baz {B1, B2, B3};
}

using name1::Baz;
using name1::B1;
using name2::B2;
typedef name1::Foo oneFoo;
typedef name1::Foo twoFoo;
Foo getFoo();
Bar getBar();

void test () {
  Foo x = FooA;
  Bar y = BarD;
  Baz z = name1::B3;
  name1::Foo a;
  oneFoo b;
  twoFoo c;
  TD td;

  while (x == FooA);
  while (y == BarD);
  while (a == name1::F1);
  while (z == name1::B2);
  while (a == b);
  while (a == c);
  while (b == c);
  while (B1 == name1::B2);
  while (B2 == name2::B1);
#ifndef MSABI
  while (x == AnonAA); // expected-warning {{comparison of constant 'AnonAA' (42) with expression of type 'Foo' is always false}}
  while (AnonBB == y); // expected-warning {{comparison of constant 'AnonBB' (45) with expression of type 'Bar' is always false}}
#endif
  while (AnonAA == AnonAB);
  while (AnonAB == AnonBA);
  while (AnonBB == AnonAA);

  while ((x) == FooA);
  while ((y) == BarD);
  while ((a) == name1::F1);
  while (z == (name1::B2));
  while (a == (b));
  while (a == (c));
  while ((b) == (c));
  while ((B1) == (name1::B2));
  while ((B2) == (name2::B1));

  while (((x)) == FooA);
  while ((y) == (BarD));
  while ((a) == (name1::F1));
  while (z == (name1::B2));
  while ((a) == ((((b)))));
  while (((a)) == (c));
  while ((b) == (((c))));
  while ((((((B1))))) == (((name1::B2))));
  while (B2 == ((((((name2::B1)))))));

  while (td == Anon1);
#ifndef MSABI
  while (td == AnonAA);  // expected-warning {{comparison of constant 'AnonAA' (42) with expression of type 'TD' is always false}}
#endif

  while (B1 == B2); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}
  while (name1::B2 == name2::B3); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}
  while (z == name2::B2); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}

  while (((((B1)))) == B2); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}
  while (name1::B2 == (name2::B3)); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}
  while (z == ((((name2::B2))))); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}

  while ((((B1))) == (((B2)))); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}
  while ((name1::B2) == (((name2::B3)))); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}
  while ((((z))) == (name2::B2)); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'name2::Baz')}}

  while (x == a); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'name1::Foo')}}
  while (x == b); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'oneFoo' (aka 'name1::Foo'))}}
  while (x == c); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'twoFoo' (aka 'name1::Foo'))}}

  while (x == y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x != y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x >= y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x <= y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x > y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x < y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (FooB == y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB != y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB >= y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB <= y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB > y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB < y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (FooB == BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB != BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB >= BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB <= BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB > BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB < BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (x == BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x != BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x >= BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x <= BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x > BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x < BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (getFoo() == y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() != y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() >= y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() <= y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() > y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() < y); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (getFoo() == BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() != BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() >= BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() <= BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() > BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() < BarD); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (getFoo() == getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() != getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() >= getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() <= getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() > getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (getFoo() < getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (FooB == getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB != getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB >= getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB <= getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB > getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (FooB < getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}

  while (x == getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x != getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x >= getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x <= getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x > getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}
  while (x < getBar()); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'Bar')}}



  while (y == x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y != x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y >= x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y <= x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y > x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y < x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (y == FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y != FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y >= FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y <= FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y > FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y < FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (BarD == FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD != FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD >= FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD <= FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD > FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD <FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (BarD == x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD != x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD >= x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD <= x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD < x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD > x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (y == getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y != getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y >= getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y <= getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y > getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (y < getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (BarD == getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD != getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD >= getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD <= getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD > getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (BarD < getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (getBar() == getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() != getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() >= getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() <= getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() > getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() < getFoo()); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (getBar() == FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() != FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() >= FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() <= FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() > FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() < FooB); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (getBar() == x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() != x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() >= x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() <= x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() > x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}
  while (getBar() < x); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'Foo')}}

  while (td == FooA); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'Foo')}}
  while (td == BarD); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'Bar')}}
  while (name1::F1 == td); // expected-warning  {{comparison of two values with different enumeration types ('name1::Foo' and 'TD')}}
  while (name2::B1 == td); // expected-warning  {{comparison of two values with different enumeration types ('name2::Baz' and 'TD')}}
  while (td == a); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'name1::Foo')}}
  while (td == b); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'oneFoo' (aka 'name1::Foo'))}}
  while (td == c); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'twoFoo' (aka 'name1::Foo'))}}
  while (td == x); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'Foo')}}
  while (td == y); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'Bar')}}
  while (td == z); // expected-warning  {{comparison of two values with different enumeration types ('TD' and 'name1::Baz')}}

  while (a == TD1); // expected-warning  {{comparison of two values with different enumeration types ('name1::Foo' and 'TD')}}
  while (b == TD2); // expected-warning  {{comparison of two values with different enumeration types ('oneFoo' (aka 'name1::Foo') and 'TD')}}
  while (c == TD1); // expected-warning  {{comparison of two values with different enumeration types ('twoFoo' (aka 'name1::Foo') and 'TD')}}
  while (x == TD2); // expected-warning  {{comparison of two values with different enumeration types ('Foo' and 'TD')}}
  while (y == TD1); // expected-warning  {{comparison of two values with different enumeration types ('Bar' and 'TD')}}
  while (z == TD2); // expected-warning  {{comparison of two values with different enumeration types ('name1::Baz' and 'TD')}}

  switch (a) {
    case name1::F1: break;
    case name1::F3: break;
    case name2::B2: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('name1::Foo' and 'name2::Baz')}}
  }

  switch (x) {
    case FooB: break;
    case FooC: break;
    case BarD: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('Foo' and 'Bar')}}
  }

  switch(getBar()) {
    case BarE: break;
    case BarF: break;
    case FooA: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('Bar' and 'Foo')}}
  }

  switch(x) {
    case AnonAA: break; // expected-warning {{case value not in enumerated type 'Foo'}}
    case FooA: break;
    case FooB: break;
    case FooC: break;
  }

  switch (td) {
    case TD1: break;
    case FooB: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('TD' and 'Foo')}}
    case BarF: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('TD' and 'Bar')}}
    // expected-warning@-1 {{case value not in enumerated type 'TD'}}
    case AnonAA: break; // expected-warning {{case value not in enumerated type 'TD'}}
  }

  switch (td) {
    case Anon1: break;
    case TD2: break;
  }

  switch (a) {
    case TD1: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('name1::Foo' and 'TD')}}
    case TD2: break; // expected-warning {{comparison of two values with different enumeration types in switch statement ('name1::Foo' and 'TD')}}
    case name1::F3: break;
  }
}
