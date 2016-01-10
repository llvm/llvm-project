// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0

void test1(pipe int *p){// expected-error {{pipes packet types cannot be of reference type}}
}
void test2(pipe p){// expected-error {{missing actual type specifier for pipe}}
}
void test3(int pipe p){// expected-error {{cannot combine with previous 'int' declaration specifier}}
}
