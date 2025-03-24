// RUN: %clang_cc1 %s -fsyntax-only -std=c++11 -verify=expected,until-cxx20
// RUN: %clang_cc1 %s -fsyntax-only -std=c++20 -verify

void f() {
    new int[]; // expected-error {{array size must be specified in new expression with no initializer}}
    new int[](); // expected-error {{cannot determine allocated array size from initializer}}
    int* a = new int[](1, 2); // until-cxx20-error {{array 'new' cannot have initialization arguments}}
    int* b = new int[] {};
    int* c = new int[] {1, 2};

    int* d = new (int[])(1, 2); // until-cxx20-error {{array 'new' cannot have initialization arguments}}
    int* e = new (int[]) {1, 2};

    using IA = int[];
    new IA; // expected-error {{array size must be specified in new expression with no initializer}}
    new IA(); // expected-error {{cannot determine allocated array size from initializer}}
    int* f = new IA(1, 2); // until-cxx20-error {{array 'new' cannot have initialization arguments}}
    int* g = new IA {};
    int* h = new IA {1, 2};

    extern int ia[];
    int* i = new decltype(ia)(1, 2); // until-cxx20-error {{array 'new' cannot have initialization arguments}}
    int* j = new decltype(ia) {1, 2};

    char* k = new char[]("hello");
    char* l = new char[] {"hello"};

    using C = char;
    char* m = new C[]("hello");
    char* n = new C[] {"hello"};

    using CA = char[];
    char* o = new CA("hello");
    char* p = new CA {"hello"};

    extern wchar_t wa[];
    wchar_t* q = new decltype(wa)(L"hello");
    wchar_t* r = new decltype(wa) {L"hello"};
}

template<class IA = int[], class C = char, class CA = char[]>
void g() {
    new IA; // expected-error {{array size must be specified in new expression with no initializer}}
    new IA(); // expected-error {{cannot determine allocated array size from initializer}}
    int* a = new IA(1, 2); // until-cxx20-error {{array 'new' cannot have initialization arguments}}
    int* b = new IA {1, 2};

    char* c = new C[]("hello");
    char* d = new C[] {"hello"};

    char* e = new CA("hello");
    char* f = new CA {"hello"};
}

template void g(); // expected-note {{in instantiation of function template specialization}}
