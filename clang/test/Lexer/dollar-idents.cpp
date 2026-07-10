// RUN: %clang_cc1 -std=c++11 -Wno-user-defined-literals -verify %s
// RUN: %clang_cc1 -std=c++11 -Wno-user-defined-literals -verify=no_dollar -fno-dollars-in-identifiers %s

int $;  // no_dollar-error {{expected unqualified-id}}
int $$; // no_dollar-error {{expected unqualified-id}}
int Σ$; // no_dollar-error {{expected ';' after top level declarator}}
int $Σ; // no_dollar-error {{expected unqualified-id}}

using size_t = decltype(sizeof(void *));

namespace UDL {

int operator"" $(const char *p, size_t n);  // no_dollar-error {{expected identifier}}
int operator"" $$(const char *p, size_t n); // no_dollar-error {{expected identifier}}
int operator"" Σ$(const char *p, size_t n); // no_dollar-error {{'operator""Σ' cannot be the name of a variable or data member}} \
                                                          // no_dollar-error {{expected ';' after top level declarator}}
int operator"" $Σ(const char *p, size_t n); // no_dollar-error {{expected identifier}}
int operator"" Σ(const char *p, size_t n);

}
namespace UDL2 {

int operator"" _$(unsigned long long); // expected-warning {{identifier '_$' preceded by whitespace in a literal operator declaration is deprecated}} \
                                       // no_dollar-error {{'operator""_' cannot be the name of a variable or data member}} \
                                       // no_dollar-error {{expected ';' after top level declarator}} \
                                       // no_dollar-warning {{identifier '_' preceded by whitespace in a literal operator declaration is deprecated}}
int a = 1_$;                           // no_dollar-error {{no matching literal operator for call}}

int operator"" _a$(unsigned long long); // expected-warning {{identifier '_a$' preceded by whitespace in a literal operator declaration is deprecated}} \
                                        // no_dollar-error {{expected ';' after top level declarator}} \
                                        // no_dollar-error {{'operator""_a' cannot be the name of a variable or data member}} \
                                        // no_dollar-warning {{identifier '_a' preceded by whitespace in a literal operator declaration is deprecated}}
int b = 1_a$;                           // no_dollar-error {{no matching literal operator for call}}


int operator""_c$(unsigned long long); // no_dollar-error {{'operator""_c' cannot be the name of a variable or data member}} \
                                       // no_dollar-error {{expected ';' after top level declarator}}
int c = 1_c$;                          // no_dollar-error {{no matching literal operator for call}}

}

#define CAT(X,Y,Z)CAT_(X,Y##Z)
#define CAT_(X,Y)X##Y
void GH171190(){
    int CAT(X,0,$); // no_dollar-error {{pasting formed '0$', an invalid preprocessing token}} \
                    // no_dollar-error {{expected ';' at end of declaration}}
}
