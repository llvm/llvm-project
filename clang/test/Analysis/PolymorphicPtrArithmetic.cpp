// RUN: %clang_cc1 -analyze -analyzer-checker=optin.cplusplus.PolymorphicPtrArithmetic -std=c++11 -verify -analyzer-output=text %s

struct Base {
    int x = 0;
    virtual ~Base() = default;
};

struct Derived : public Base {};

struct DoubleDerived : public Derived {};

Derived *get();

Base *create() {
    Base *b = new Derived[3]; // expected-note{{Casting from 'Derived' to 'Base' here}}
    return b;
}

void sink(Base *b) {
    b[1].x++; // expected-warning{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<Derived*>(b);
}

void sink_cast(Base *b) {
    static_cast<Derived*>(b)[1].x++; // no-warning
    delete[] static_cast<Derived*>(b);
}

void sink_derived(Derived *d) {
    d[1].x++; // no-warning
    delete[] static_cast<Derived*>(d);
}

void same_function() {
    Base *sd = new Derived[10]; // expected-note{{Casting from 'Derived' to 'Base' here}}
    // expected-note@-1{{Casting from 'Derived' to 'Base' here}}
    sd[1].x++; // expected-warning{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    (sd + 1)->x++; // expected-warning{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<Derived*>(sd);
    
    Base *dd = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    dd[1].x++; // expected-warning{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<DoubleDerived*>(dd);

}

void different_function() {
    Base *assigned = get(); // expected-note{{Casting from 'Derived' to 'Base' here}}
    assigned[1].x++; // expected-warning{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<Derived*>(assigned);

    Base *indirect;
    indirect = get(); // expected-note{{Casting from 'Derived' to 'Base' here}}
    indirect[1].x++; // expected-warning{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<Derived*>(indirect);

    Base *created = create(); // expected-note{{Calling 'create'}}
    // expected-note@-1{{Returning from 'create'}}
    created[1].x++; // expected-warning{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'Derived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<Derived*>(created);

    Base *sb = new Derived[10]; // expected-note{{Casting from 'Derived' to 'Base' here}}
    sink(sb); // expected-note{{Calling 'sink'}}
}

void safe_function() {
    Derived *d = new Derived[10];
    d[1].x++; // no-warning
    delete[] static_cast<Derived*>(d);

    Base *b = new Derived[10];
    static_cast<Derived*>(b)[1].x++; // no-warning
    delete[] static_cast<Derived*>(b);

    Base *sb = new Derived[10];
    sink_cast(sb); // no-warning

    Derived *sd = new Derived[10];
    sink_derived(sd); // no-warning
}

void multiple_derived() {
    Base *b = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    b[1].x++; // expected-warning{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<DoubleDerived*>(b);

    Base *b2 = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    Derived *d2 = static_cast<Derived*>(b2); // expected-note{{Casting from 'Base' to 'Derived' here}}
    d2[1].x++; // expected-warning{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    delete[] static_cast<DoubleDerived*>(d2);

    Derived *d3 = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Derived' here}}
    Base *b3 = d3; // expected-note{{Casting from 'Derived' to 'Base' here}}
    b3[1].x++; // expected-warning{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Base' is undefined}}
    delete[] static_cast<DoubleDerived*>(b3);

    Base *b4 = new DoubleDerived[10];
    Derived *d4 = static_cast<Derived*>(b4);
    DoubleDerived *dd4 = static_cast<DoubleDerived*>(d4);
    dd4[1].x++; // no-warning
    delete[] static_cast<DoubleDerived*>(dd4);

    Base *b5 = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    DoubleDerived *dd5 = static_cast<DoubleDerived*>(b5); // expected-note{{Casting from 'Base' to 'DoubleDerived' here}}
    Derived *d5 = dd5; // expected-note{{Casting from 'DoubleDerived' to 'Derived' here}}
    d5[1].x++; // expected-warning{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    delete[] static_cast<DoubleDerived*>(d5);
}

void other_operators() {
    Derived *d = new Derived[10];
    Base *b1 = d;
    Base *b2 = d + 1;
    Base *b3 = new Derived[10];
    if (b1 < b2) return; // no-warning
    if (b1 == b3) return; // no-warning
}

void unrelated_casts() {
    Base *b = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    Base &b2 = *b; // no-note: See the FIXME.

    // FIXME: Displaying casts of reference types is not supported.
    Derived &d2 = static_cast<Derived&>(b2); // no-note: See the FIXME.

    Derived *d = &d2; // no-note: See the FIXME.
    d[1].x++; // expected-warning{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    // expected-note@-1{{Doing pointer arithmetic with 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    delete[] static_cast<DoubleDerived*>(d);
}
