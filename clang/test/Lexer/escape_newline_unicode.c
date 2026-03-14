// RUN: %clang_cc1 -verify=expected,c -x c -Wunused %s
// RUN: %clang_cc1 -verify=expected,cpp -x c++ -Wunused %s

void gh65156(void) {

int a\
ス = 42;
// expected-warning@-2 {{unused variable 'aス'}}

int b\ 
\ 
ス = 42;
// expected-warning@-2 {{backslash and newline separated by space}}
// expected-warning@-4 {{backslash and newline separated by space}}
// expected-warning@-5 {{unused variable 'bス'}}

int ス\
ス = 42;
// expected-warning@-2 {{unused variable 'スス'}}

int \
ス = 42;
// expected-warning@-2 {{unused variable 'ス'}}

}

void gh65156_err(void) {

int \
❌ = 0;
// cpp-error@-2 {{expected unqualified-id}}
// c-error@-3 {{expected identifier}}


int a\
❌ = 0;
// expected-error@-1 {{character <U+274C> not allowed in an identifier}}
}
