// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -Wno-anonymous-pack-parens %s

namespace ParamTypes {

// Make sure we accept this
template<class X>struct A{typedef X Y;};
template<class X>bool operator==(A<X>,typename A<X>::Y); // expected-note{{candidate template ignored: could not match 'A<X>' against 'B<int> *'}}

int a(A<int> x) { return operator==(x,1); }

int a0(A<int> x) { return x == 1; }

template<class X>struct B{typedef X Y;};
template<class X>bool operator==(B<X>*,typename B<X>::Y); // expected-note{{candidate template ignored: substitution failure [with X = int]}}
int a(B<int> x) { return operator==(&x,1); } // expected-error{{no matching function for call to 'operator=='}}

} // namespace ParamTypes

namespace CompareOrdering {

// Ensure we take parameter list reversal into account in partial ordering.
template<typename T> struct A {};
template<typename T> int operator<=>(A<T>, int) = delete;
template<typename T> int operator<=>(int, A<T*>);
// OK, selects the more-specialized reversed function.
bool b = A<int*>() < 0;

} // namespace CompareOrdering

namespace SFINAE {

struct A {
    constexpr operator int() { return 1; }
};

int operator+(auto, int) { static_assert(false); }

static_assert(2 + A() == 3);


struct B {};

constexpr bool operator==(auto, int i) {
    return i;
}

static_assert(B() != 0);
static_assert(B() == 1);
static_assert(0 != B());
static_assert(1 == B());


struct C {
    int operator++(auto) { static_assert(false); }
    constexpr int operator++(int) { return 1; }
};

static_assert(C().operator++(1.0) == 1);

int operator++(C, auto) { static_assert(false); }
constexpr int operator++(C, int) { return 2; }

static_assert(operator++(C(), 1.0) == 2);


constexpr int operator%(auto...) {
    return 10;
}

static_assert(A() % B() == 10);

} // namespace SFINAE

namespace InvalidDecls {

class Bad {
    void operator~(auto);         // expected-error {{overloaded 'operator~' must be a unary operator (has 2 parameters)}}
    void operator~(auto...);      // expected-error {{overloaded 'operator~' must be a unary operator (has 2 parameters)}}
    void operator+(int, auto);    // expected-error {{overloaded 'operator+' must be a unary or binary operator (has 3 parameters)}}
    void operator+(int, auto...); // expected-error {{overloaded 'operator+' must be a unary or binary operator (has 3 parameters)}}
    void operator/(int, auto);    // expected-error {{overloaded 'operator/' must be a binary operator (has 3 parameters)}}
    void operator/(int, auto...); // expected-error {{overloaded 'operator/' must be a binary operator (has 3 parameters)}}
};

void operator~(Bad, auto);         // expected-error {{overloaded 'operator~' must be a unary operator (has 2 parameters)}}
void operator~(Bad, auto...);      // expected-error {{overloaded 'operator~' must be a unary operator (has 2 parameters)}}
void operator+(Bad, int, auto);    // expected-error {{overloaded 'operator+' must be a unary or binary operator (has 3 parameters)}}
void operator+(Bad, int, auto...); // expected-error {{overloaded 'operator+' must be a unary or binary operator (has 3 parameters)}}
void operator/(auto);              // expected-error {{overloaded 'operator/' must be a binary operator (has 1 parameter)}}
void operator/(Bad, int, auto);    // expected-error {{overloaded 'operator/' must be a binary operator (has 3 parameters)}}
void operator/(Bad, int, auto...); // expected-error {{overloaded 'operator/' must be a binary operator (has 3 parameters)}}

class C;
int operator+(auto*);         // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator-(auto[]);        // expected-error {{overloaded 'operator-' must have at least one parameter of class or enumeration type}}
int operator+(auto());        // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto C::*);     // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto*&);        // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto (&)[]);    // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto (&)());    // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto C::*&);    // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto*...);      // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator-(auto... _[]);   // expected-error {{overloaded 'operator-' must have at least one parameter of class or enumeration type}}
int operator+(auto...());     // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto C::*...);  // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto*&...);     // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto (&...)[]); // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto (&...)()); // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}
int operator+(auto C::*&...); // expected-error {{overloaded 'operator+' must have at least one parameter of class or enumeration type}}

int operator++(auto);
int operator++(auto...);
int operator++(auto, auto);
int operator++(auto*...);         // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator++(auto*, auto);      // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator--(auto[], auto);     // expected-error {{overloaded 'operator--' must have at least one parameter of class or enumeration type}}
int operator++(auto(), auto);     // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator++(auto C::*, auto);  // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator++(auto*&, auto);     // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator++(auto (&)[], auto); // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator++(auto (&)(), auto); // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}
int operator++(auto C::*&, auto); // expected-error {{overloaded 'operator++' must have at least one parameter of class or enumeration type}}

int operator~(auto, ...);       // expected-error {{overloaded 'operator~' cannot be variadic}}
int operator~(auto..., ...);    // expected-error {{overloaded 'operator~' cannot be variadic}}
int operator/(auto..., ...);    // expected-error {{overloaded 'operator/' cannot be variadic}}
int operator/(auto, auto, ...); // expected-error {{overloaded 'operator/' cannot be variadic}}

int operator!(auto...); // expected-note-re 2 {{candidate template ignored: substitution failure [with {{.+}}]: overloaded 'operator!' must be a unary operator}}
int bad1 = operator!();             // expected-error {{no matching function for call to 'operator!'}}
int bad2 = operator!(Bad(), Bad()); // expected-error {{no matching function for call to 'operator!'}}

int operator*(auto...); // expected-note-re 2 {{candidate template ignored: substitution failure [with {{.+}}]: overloaded 'operator*' must be a unary or binary operator}}
int bad3 = operator*();                    // expected-error {{no matching function for call to 'operator*'}}
int bad4 = operator*(Bad(), Bad(), Bad()); // expected-error {{no matching function for call to 'operator*'}}

int operator%(auto...); // expected-note-re 3 {{candidate template ignored: substitution failure [with {{.+}}]: overloaded 'operator%' must be a binary operator}}
int bad5 = operator%();                    // expected-error {{no matching function for call to 'operator%'}}
int bad6 = operator%(Bad());               // expected-error {{no matching function for call to 'operator%'}}
int bad7 = operator%(Bad(), Bad(), Bad()); // expected-error {{no matching function for call to 'operator%'}}

int operator&(auto...); // expected-note-re 4 {{candidate template ignored: substitution failure [with {{.+}}]: overloaded 'operator&' must}}

template<> int operator&();              // expected-error {{no function template matches function template specialization 'operator&'}}
template<> int operator&(int);           // expected-error {{no function template matches function template specialization 'operator&'}}
template<> int operator&<int>(int);      // expected-error {{no function template matches function template specialization 'operator&'}}
template<> int operator&(Bad, Bad, Bad); // expected-error {{no function template matches function template specialization 'operator&'}}

int operator-(auto...); // expected-note-re 4 {{candidate template ignored: substitution failure [with {{.+}}]: overloaded 'operator-' must}}

template int operator-();              // expected-error {{explicit instantiation of 'operator-' does not refer to a function template}}
template int operator-(int);           // expected-error {{explicit instantiation of 'operator-' does not refer to a function template}}
template int operator-<int>(int);      // expected-error {{explicit instantiation of 'operator-' does not refer to a function template}}
template int operator-(Bad, Bad, Bad); // expected-error {{explicit instantiation of 'operator-' does not refer to a function template}}

template<class... Ts>
class F {
    friend void operator^(Ts...); // expected-error 3 {{overloaded 'operator^' must be a binary operator}}
};

F<> s1;              // expected-note {{in instantiation of template class}}
F<Bad> s2;           // expected-note {{in instantiation of template class}}
F<Bad, Bad> s3;
F<Bad, Bad, Bad> s4; // expected-note {{in instantiation of template class}}

} // namespace InvalidDecls

namespace BadConstraints {

template<class...>
int Poison;

struct S {
    S(int);
    operator int();
};

template<class T>
int operator+(T, int) requires Poison<T>; // expected-error {{atomic constraint must be of type 'bool'}}
int operator+(int, auto);

int i = 1 + S(2);
// expected-note@-1 {{while checking constraint satisfaction for template 'operator+<int>' required here}}
// expected-note@-2 {{in instantiation of function template specialization 'BadConstraints::operator+<int>' requested here}}

} // namespace BadConstraints
