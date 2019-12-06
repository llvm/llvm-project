// RUN: %clang_cc1 -std=c++2a -fconcepts-ts -x c++ -verify %s

template<typename T> requires sizeof(T) >= 2 // expected-note{{because 'sizeof(char) >= 2' (1 >= 2) evaluated to false}}
struct A {
  static constexpr int value = sizeof(T);
};

static_assert(A<int>::value == 4);
static_assert(A<char>::value == 1); // expected-error{{constraints not satisfied for class template 'A' [with T = char]}}

template<typename T, typename U>
  requires sizeof(T) != sizeof(U) // expected-note{{because 'sizeof(int) != sizeof(char [4])' (4 != 4) evaluated to false}}
           && sizeof(T) >= 4 // expected-note{{because 'sizeof(char) >= 4' (1 >= 4) evaluated to false}}
constexpr int SizeDiff = sizeof(T) > sizeof(U) ? sizeof(T) - sizeof(U) : sizeof(U) - sizeof(T);

static_assert(SizeDiff<int, char> == 3);
static_assert(SizeDiff<int, char[4]> == 0); // expected-error{{constraints not satisfied for variable template 'SizeDiff' [with T = int, U = char [4]]}}
static_assert(SizeDiff<char, int> == 3); // expected-error{{constraints not satisfied for variable template 'SizeDiff' [with T = char, U = int]}}

template<typename... Ts>
  requires ((sizeof(Ts) == 4) || ...) // expected-note{{because 'sizeof(char) == 4' (1 == 4) evaluated to false}} expected-note{{'sizeof(long long) == 4' (8 == 4) evaluated to false}} expected-note{{'sizeof(int [20]) == 4' (80 == 4) evaluated to false}}
constexpr auto SumSizes = (sizeof(Ts) + ...);

static_assert(SumSizes<char, long long, int> == 13);
static_assert(SumSizes<char, long long, int[20]> == 89); // expected-error{{constraints not satisfied for variable template 'SumSizes' [with Ts = <char, long long, int [20]>]}}

template<typename T>
concept IsBig = sizeof(T) > 100; // expected-note{{because 'sizeof(int) > 100' (4 > 100) evaluated to false}}

template<typename T>
  requires IsBig<T> // expected-note{{'int' does not satisfy 'IsBig'}}
using BigPtr = T*;

static_assert(sizeof(BigPtr<int>)); // expected-error{{constraints not satisfied for alias template 'BigPtr' [with T = int]}}}}

template<typename T> requires T::value // expected-note{{because substituted constraint expression is ill-formed: type 'int' cannot be used prior to '::' because it has no members}}
struct S { static constexpr bool value = true; };

struct S2 { static constexpr bool value = true; };

static_assert(S<int>::value); // expected-error{{constraints not satisfied for class template 'S' [with T = int]}}
static_assert(S<S2>::value);

template<typename T>
struct AA
{
    template<typename U> requires sizeof(U) == sizeof(T) // expected-note{{because 'sizeof(int [2]) == sizeof(int)' (8 == 4) evaluated to false}}
    struct B
    {
        static constexpr int a = 0;
    };

    template<typename U> requires sizeof(U) == sizeof(T) // expected-note{{because 'sizeof(int [2]) == sizeof(int)' (8 == 4) evaluated to false}}
    static constexpr int b = 1;

    template<typename U> requires sizeof(U) == sizeof(T) // expected-note{{because 'sizeof(int [2]) == sizeof(int)' (8 == 4) evaluated to false}}
    static constexpr int getB() { // expected-note{{candidate template ignored: constraints not satisfied [with U = int [2]]}}
        return 2;
    }

    static auto foo()
    {
        return B<T[2]>::a; // expected-error{{constraints not satisfied for class template 'B' [with U = int [2]]}}
    }

    static auto foo1()
    {
        return b<T[2]>; // expected-error{{constraints not satisfied for variable template 'b' [with U = int [2]]}}
    }

    static auto foo2()
    {
        return AA<T>::getB<T[2]>(); // expected-error{{no matching function for call to 'getB'}}
    }
};

constexpr auto x = AA<int>::foo(); // expected-note{{in instantiation of member function 'AA<int>::foo' requested here}}
constexpr auto x1 = AA<int>::foo1(); // expected-note{{in instantiation of member function 'AA<int>::foo1' requested here}}
constexpr auto x2 = AA<int>::foo2(); // expected-note{{in instantiation of member function 'AA<int>::foo2' requested here}}

template<typename T>
struct B { using type = typename T::type; }; // expected-error{{type 'int' cannot be used prior to '::' because it has no members}}

template<typename T> requires B<T>::type // expected-note{{in instantiation of template class 'B<int>' requested here}}
                                         // expected-note@-1{{while substituting template arguments into constraint expression here}}
struct C { };

template<typename T> requires T{} // expected-error{{atomic constraint must be of type 'bool' (found 'int')}}
struct D { };

static_assert(C<int>{}); // expected-note{{while checking constraint satisfaction for template 'C<int>' required here}}
static_assert(D<int>{}); // expected-note{{while checking constraint satisfaction for template 'D<int>' required here}}