// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %s -Wpre-c23-compat
// RUN: %clang_cc1 -fsyntax-only -verify=c17 -std=c17 %s -Wc23-compat

constexpr int a = 0; // c17-warning {{'constexpr' is a keyword in C23}} \
                        c17-error {{unknown type name 'constexpr'}} \
                        c23-warning {{'constexpr' specifier is incompatible with C standards before C23}}

void func(int array[constexpr]); // c23-error {{expected expression}} \
                                 // c17-error {{use of undeclared}}

_Atomic constexpr int b = 0; // c23-error {{constexpr variable cannot have type 'const _Atomic(int)'}} \
                             // c23-warning {{'constexpr' specifier is incompatible with C standards before C23}} \
                             // c17-error {{unknown type name 'constexpr'}}

int static constexpr c = 1; // c17-error {{expected ';' after top level declarator}} \
                            // c23-warning {{'constexpr' specifier is incompatible with C standards before C23}}

struct constexpr {  // c23-error {{declaration of anonymous struct must be a definition}} \
                       c23-warning {{declaration does not declare anything}}
    int constexpr;
};

struct constexpr f(void) {  // c23-error {{declaration of anonymous struct must be a definition}}
    typedef int constexpr;
    struct constexpr c;
    c.constexpr = (constexpr) 0;
    return c;
}  // c23-error {{expected identifier or '('}}
