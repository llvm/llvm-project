// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST1
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST2
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST3
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST4
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST5
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST6
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST7
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST8

// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ %s -std=c++11 -fsyntax-only -verify -DTEST9
// RUN: not %clang_cc1 -x c++ %t -std=c++11 -fixit -DTEST9
// RUN: %clang_cc1 -x c++ %t -std=c++11 -fsyntax-only -DTEST9

// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST10
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST11
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST12
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST13
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST14

#if TEST1

// expected-no-diagnostics
typedef int Int;
typedef char Char;
typedef Char* Carp;

Int main(Int argc, Carp argv[]) {
}

#elif TEST2

// expected-no-diagnostics
typedef int Int;
typedef char Char;
typedef Char* Carp;

Int main(Int argc, Carp argv[], Char *env[]) {
}

#elif TEST3

// expected-no-diagnostics
int main() {
}

#elif TEST4

static int main() { // expected-error {{'main' is not allowed to be declared static}}
}

#elif TEST5

inline int main() { // expected-error {{'main' is not allowed to be declared inline}}
}

#elif TEST6

void  // expected-error {{'main' must return 'int'}}
main( // expected-error {{first parameter of 'main' (argument count) must be of type 'int'}}
     float a
) {
}

const int main(); // expected-error {{'main' must return 'int'}}

#elif TEST7

// expected-no-diagnostics
int main(int argc, const char* const* argv) {
}

#elif TEST8

template<typename T>
int main() { } // expected-error{{'main' cannot be a template}}

#elif TEST9

constexpr int main() { } // expected-error{{'main' is not allowed to be declared constexpr}}

#elif TEST10

// PR15100
// expected-no-diagnostics
typedef char charT;
int main(int, const charT**) {}

#elif TEST11

// expected-no-diagnostics
typedef char charT;
int main(int, charT* const *) {}

#elif TEST12

// expected-no-diagnostics
typedef char charT;
int main(int, const charT* const *) {}

#elif TEST13

int main(void) {}

template <typename T>
int main(void); // expected-error{{'main' cannot be a template}}

#elif TEST14

template <typename T>
int main(void); // expected-error{{'main' cannot be a template}}

int main(void) {}

#else

#error Unknown test mode

#endif
