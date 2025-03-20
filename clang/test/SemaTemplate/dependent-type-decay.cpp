// RUN: %clang_cc1 %s -fsyntax-only -std=c++20 -verify

void f(auto*) {} // expected-note {{previous definition is here}}
void f(auto[]) {} // expected-error {{redefinition of 'f'}}

void g(auto()) {} // expected-note {{previous definition is here}}
void g(auto (*)()) {} // expected-error {{redefinition of 'g'}}

void fp(auto*...) {} // expected-note {{previous definition is here}}
void fp(auto... _[]) {} // expected-error {{redefinition of 'fp'}}

void gp(auto...()) {} // expected-note {{previous definition is here}}
void gp(auto (*..._)()) {} // expected-error {{redefinition of 'gp'}}


template<int*> class A;
template<int _[]> class A;

template<int()> class B;
template<int (*)()> class B;

template<int*...> class Ap;
template<int... _[]> class Ap;

template<int...()> class Bp;
template<int (*..._)()> class Bp;

template<class T, class... Ts>
class C {
    template<T*> void f(); // expected-note {{previous declaration is here}}
    template<T[]> void f(); // expected-error {{class member cannot be redeclared}}

    template<T()> void g(); // expected-note {{previous declaration is here}}
    template<T(*)()> void g(); // expected-error {{class member cannot be redeclared}}

    template<Ts*...> void fp(); // expected-note {{previous declaration is here}}
    template<Ts... _[]> void fp(); // expected-error {{class member cannot be redeclared}}

    template<Ts...()> void gp(); // expected-note {{previous declaration is here}}
    template<Ts(* ..._)()> void gp(); // expected-error {{class member cannot be redeclared}}

    template<T[]> class X;
    template<T()> class Y;

    template<Ts... _[]> class Xp;
    template<Ts...()> class Yp;
};

template<class T, class... Ts>
template<T*>
class C<T, Ts...>::X {};

template<class T, class... Ts>
template<T (*)()>
class C<T, Ts...>::Y {};

template<class T, class... Ts>
template<Ts*...>
class C<T, Ts...>::Xp {};

template<class T, class... Ts>
template<Ts (*..._)()>
class C<T, Ts...>::Yp {};


template<class...> class R;

template<class T>
R<T>* d0(T[]) { return 0; }
R<int>* r0 = d0((int*)0);

template<class T>
R<T>* d1(T(T[])) { return 0; }
R<int>* r1 = d1((int (*)(int*))0);

template<class T>
R<T>* d2(T(T(T))) { return 0; }
R<int>* r2 = d2((int (*)(int (*)(int)))0);

template<class... Ts>
R<Ts...>* d0p(Ts... _[]) { return 0; }
R<int, char>* r0p = d0p((int*)0, (char*)0);

template<class... Ts>
R<Ts...>* d1p(Ts...(Ts... _[])) { return 0; }
R<int, char>* r1p = d1p((int (*)(int*, char*))0, (char (*)(int*, char*))0);

template<class... Ts>
R<Ts...>* d2p(Ts...(Ts...(Ts...))) { return 0; }
R<int, char>* r2p = d2p(
    (int (*)(int(int, char), char(int, char)))0,
    (char (*)(int(int, char), char(int, char)))0);


template<class T> concept Y = sizeof(T*) != 0;

template<class T, class... Ts>
struct S {
    static int f() requires Y<int(T*)>;
    static constexpr int f() requires Y<int(T[1])> && true {
        return 1;
    }

    static int g() requires Y<int(T (*)())>;
    static constexpr int g() requires Y<int(T())> && true {
        return 2;
    }

    static int fp() requires Y<int(Ts*...)>;
    static constexpr int fp() requires Y<int(Ts... _[1])> && true {
        return 3;
    }

    static int gp() requires Y<int(Ts (*..._)(Ts...))>;
    static constexpr int gp() requires Y<int(Ts...(Ts...))> && true {
        return 4;
    }
};

static_assert(S<char, short, int>::f() == 1);
static_assert(S<char, short, int>::g() == 2);
static_assert(S<char, short, int>::fp() == 3);
static_assert(S<char, short, int>::gp() == 4);
