// RUN: %clang_cc1 %s -fsyntax-only -verify 

typedef struct __CFError * CFErrorRef; // expected-note {{forward declaration of 'struct __CFError'}}

void junk(int, ...);

int main(void)
{
 CFErrorRef error;
 junk(1, *error, (void)0); // expected-error {{argument type 'struct __CFError' is incomplete}} \
                           // expected-error {{argument type 'void' is incomplete}}
}
