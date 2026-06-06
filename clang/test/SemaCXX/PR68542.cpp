// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s -fexperimental-new-constant-interpreter

struct S { // expected-note {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'S &&' for 1st argument}} \
           // expected-note {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const S &' for 1st argument}}
    int e;
};

template<class T>
consteval int get_format() {
	return nullptr; // expected-error {{cannot initialize return object of type 'int' with an rvalue of type 'std::nullptr_t'}}
}

template<class T>
constexpr S f(T) noexcept {
	return get_format<T>(); // expected-error {{no viable conversion from returned value of type 'int' to function return type 'S'}}
}

constexpr S x = f(0); // expected-error {{constexpr variable 'x' must be initialized by a constant expression}} \
                      // expected-note {{in instantiation of function template specialization 'f<int>' requested here}}
