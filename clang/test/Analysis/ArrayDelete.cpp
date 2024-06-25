// RUN: %clang_cc1 -analyze -analyzer-checker=cplusplus.ArrayDelete -std=c++11 -verify -analyzer-output=text %s

struct Base {
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
    delete[] b; // expected-warning{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
}

void sink_cast(Base *b) {
    delete[] static_cast<Derived*>(b); // no-warning
}

void sink_derived(Derived *d) {
    delete[] d; // no-warning
}

void same_function() {
    Base *sd = new Derived[10]; // expected-note{{Casting from 'Derived' to 'Base' here}}
    delete[] sd; // expected-warning{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
    
    Base *dd = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    delete[] dd; // expected-warning{{Deleting an array of 'DoubleDerived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'DoubleDerived' objects as their base class 'Base' is undefined}}
}

void different_function() {
    Base *assigned = get(); // expected-note{{Casting from 'Derived' to 'Base' here}}
    delete[] assigned; // expected-warning{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}

    Base *indirect;
    indirect = get(); // expected-note{{Casting from 'Derived' to 'Base' here}}
    delete[] indirect; // expected-warning{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}

    Base *created = create(); // expected-note{{Calling 'create'}}
    // expected-note@-1{{Returning from 'create'}}
    delete[] created; // expected-warning{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'Derived' objects as their base class 'Base' is undefined}}

    Base *sb = new Derived[10]; // expected-note{{Casting from 'Derived' to 'Base' here}}
    sink(sb); // expected-note{{Calling 'sink'}}
}

void safe_function() {
    Derived *d = new Derived[10];
    delete[] d; // no-warning

    Base *b = new Derived[10];
    delete[] static_cast<Derived*>(b); // no-warning

    Base *sb = new Derived[10];
    sink_cast(sb); // no-warning

    Derived *sd = new Derived[10];
    sink_derived(sd); // no-warning
}

void multiple_derived() {
    Base *b = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    delete[] b; // expected-warning{{Deleting an array of 'DoubleDerived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'DoubleDerived' objects as their base class 'Base' is undefined}}

    Base *b2 = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    Derived *d2 = static_cast<Derived*>(b2); // expected-note{{Casting from 'Base' to 'Derived' here}}
    delete[] d2; // expected-warning{{Deleting an array of 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    // expected-note@-1{{Deleting an array of 'DoubleDerived' objects as their base class 'Derived' is undefined}}

    Derived *d3 = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Derived' here}}
    Base *b3 = d3; // expected-note{{Casting from 'Derived' to 'Base' here}}
    delete[] b3; // expected-warning{{Deleting an array of 'DoubleDerived' objects as their base class 'Base' is undefined}}
    // expected-note@-1{{Deleting an array of 'DoubleDerived' objects as their base class 'Base' is undefined}}

    Base *b4 = new DoubleDerived[10];
    Derived *d4 = static_cast<Derived*>(b4);
    DoubleDerived *dd4 = static_cast<DoubleDerived*>(d4);
    delete[] dd4; // no-warning

    Base *b5 = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    DoubleDerived *dd5 = static_cast<DoubleDerived*>(b5); // expected-note{{Casting from 'Base' to 'DoubleDerived' here}}
    Derived *d5 = dd5; // expected-note{{Casting from 'DoubleDerived' to 'Derived' here}}
    delete[] d5; // expected-warning{{Deleting an array of 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    // expected-note@-1{{Deleting an array of 'DoubleDerived' objects as their base class 'Derived' is undefined}}
}

void unrelated_casts() {
    Base *b = new DoubleDerived[10]; // expected-note{{Casting from 'DoubleDerived' to 'Base' here}}
    Base &b2 = *b; // no-note: See the FIXME.

    // FIXME: Displaying casts of reference types is not supported.
    Derived &d2 = static_cast<Derived&>(b2); // no-note: See the FIXME.

    Derived *d = &d2; // no-note: See the FIXME.
    delete[] d; // expected-warning{{Deleting an array of 'DoubleDerived' objects as their base class 'Derived' is undefined}}
    // expected-note@-1{{Deleting an array of 'DoubleDerived' objects as their base class 'Derived' is undefined}}
}
