// RUN: %clang_cc1 -fsyntax-only -std=c++2c -verify=expected,cxx26 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++2b -verify=expected,cxx23 %s


struct S {};
struct T : S {} t;

consteval void test() {
    void* a = &t;
    const void* b = &t;
    volatile void* c = &t;
    (void)static_cast<T*>(a);
    (void)static_cast<const T*>(a);
    (void)static_cast<volatile T*>(a);

    (void)(T*)(a);
    (void)(const T*)(a);
    (void)(volatile T*)(a);

    (void)static_cast<T*>(b); // expected-error {{static_cast from 'const void *' to 'T *' casts away qualifiers}}
    (void)static_cast<volatile T*>(b); // expected-error {{static_cast from 'const void *' to 'volatile T *' casts away qualifiers}}
    (void)static_cast<const T*>(b);
    (void)static_cast<volatile const T*>(b);

    (void)static_cast<T*>(c); // expected-error{{static_cast from 'volatile void *' to 'T *' casts away qualifiers}}
    (void)static_cast<volatile T*>(c);
    (void)static_cast<const T*>(b);
    (void)static_cast<volatile const T*>(b);
}

void err() {
    constexpr void* a = &t;
    constexpr auto err1 = static_cast<int*>(a); // expected-error{{constexpr variable 'err1' must be initialized by a constant expression}} \
                                                // cxx23-note {{cast from 'void *' is not allowed in a constant expression in C++ standards before C++2c}} \
                                                // cxx26-note {{cast from 'void *' is not allowed in a constant expression because the pointed object type 'T' is not similar to the target type 'int'}}
    constexpr auto err2 = static_cast<S*>(a);   // expected-error{{constexpr variable 'err2' must be initialized by a constant expression}} \
                                                // cxx23-note {{cast from 'void *' is not allowed in a constant expression in C++ standards before C++2c}} \
                                                // cxx26-note {{cast from 'void *' is not allowed in a constant expression because the pointed object type 'T' is not similar to the target type 'S'}}
}

int* p;
constexpr int** pp = &p;
constexpr void* vp = pp;
constexpr auto cvp = static_cast<const int* volatile*>(vp);
// cxx23-error@-1 {{constant expression}}
// cxx23-note@-2 {{cast from 'void *' is not allowed in a constant expression}}
