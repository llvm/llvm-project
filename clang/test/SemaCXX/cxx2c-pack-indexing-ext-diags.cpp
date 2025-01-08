// RUN: %clang_cc1 -std=c++2c -verify=cxx26 -fsyntax-only -Wpre-c++26-compat %s
// RUN: %clang_cc1 -std=c++11 -verify=cxx11 -fsyntax-only -Wc++26-extensions %s

template <typename... T>
void f(T... t) {
    // cxx26-warning@+2 {{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 {{pack indexing is a C++2c extension}}
    using a = T...[0];

    // cxx26-warning@+2 {{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 {{pack indexing is a C++2c extension}}
    using b = typename T...[0]::a;

    // cxx26-warning@+2 2{{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 2{{pack indexing is a C++2c extension}}
    t...[0].~T...[0]();

    // cxx26-warning@+2 {{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 {{pack indexing is a C++2c extension}}
    T...[0] c;
}

template <typename... T>
void g(T... [1]); // cxx11-warning {{'T...[1]' is no longer a pack expansion but a pack indexing type; add a name to specify a pack expansion}} \
                  // cxx11-warning {{pack indexing is a C++2c extension}} \
                  // cxx11-note {{candidate function template not viable}} \
                  // cxx26-warning {{pack indexing is incompatible with C++ standards before C++2c}} \
                  // cxx26-note {{candidate function template not viable}}

template <typename... T>
void h(T... param[1]);

template <class T>
struct S {
  using type = T;
};

template <typename... T>
void h(typename T... [1]::type); // cxx11-warning {{pack indexing is a C++2c extension}} \
                                 // cxx26-warning {{pack indexing is incompatible with C++ standards before C++2c}}

template <typename... T>
void x(T... [0]); // cxx11-warning {{'T...[0]' is no longer a pack expansion but a pack indexing type; add a name to specify a pack expansion}} \
                  // cxx11-warning {{pack indexing is a C++2c extension}} \
                  // cxx26-warning {{pack indexing is incompatible with C++ standards before C++2c}}

void call() {
  g<int, double>(nullptr, nullptr); // cxx26-error {{no matching function for call to 'g'}} \
                                    // cxx11-error {{no matching function for call to 'g'}}
  h<int, double>(nullptr, nullptr);
  h<S<int>, S<const char *>>("hello");
  x<int*>(nullptr);
}
