// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

struct ValueInt
{
  ValueInt(int v = 0) : ValueLength(v) {}
  operator int () const { return ValueLength; } // expected-note 3{{conversion to integral type 'int' declared here}}
private:
  int ValueLength;
};

enum E { e };
struct ValueEnum {
  operator E() const; // expected-note{{conversion to enumeration type 'E' declared here}}
};

struct ValueBoth : ValueInt, ValueEnum { };

struct IndirectValueInt : ValueInt { };
struct TwoValueInts : ValueInt, IndirectValueInt { }; // expected-warning{{direct base 'ValueInt' is inaccessible due to ambiguity:\n    struct TwoValueInts -> struct ValueInt\n    struct TwoValueInts -> struct IndirectValueInt -> struct ValueInt}}


void test() {
  (void)new int[ValueInt(10)]; // expected-warning{{implicit conversion from array size expression of type 'ValueInt' to integral type 'int' is a C++11 extension}}
  (void)new int[ValueEnum()]; // expected-warning{{implicit conversion from array size expression of type 'ValueEnum' to enumeration type 'E' is a C++11 extension}}
  (void)new int[ValueBoth()]; // expected-error{{ambiguous conversion of array size expression of type 'ValueBoth' to an integral or enumeration type}}

  (void)new int[TwoValueInts()]; // expected-error{{ambiguous conversion of array size expression of type 'TwoValueInts' to an integral or enumeration type}}
}
