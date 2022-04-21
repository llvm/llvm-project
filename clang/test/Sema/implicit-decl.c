// RUN: %clang_cc1 %s -verify -fsyntax-only -Werror=implicit-function-declaration -std=c99
// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c11
// RUN: %clang_cc1 %s -verify=c2x -fsyntax-only -std=c2x

/// -Werror-implicit-function-declaration is a deprecated alias used by many projects.
// RUN: %clang_cc1 %s -verify -fsyntax-only -Werror-implicit-function-declaration

// c2x-note@*:* {{'__builtin_va_list' declared here}}

typedef int int32_t;
typedef unsigned char Boolean;

extern int printf(__const char *__restrict __format, ...); // expected-note{{'printf' declared here}} \
                                                              c2x-note {{'printf' declared here}}

void func(void) {
   int32_t *vector[16];
   const char compDesc[16 + 1];
   int32_t compCount = 0;
   if (_CFCalendarDecomposeAbsoluteTimeV(compDesc, vector, compCount)) { // expected-error {{call to undeclared function '_CFCalendarDecomposeAbsoluteTimeV'; ISO C99 and later do not support implicit function declarations}} \
                                                                            expected-note {{previous implicit declaration}} \
                                                                            c2x-error {{use of undeclared identifier '_CFCalendarDecomposeAbsoluteTimeV'}}
   }

   printg("Hello, World!\n"); // expected-error{{call to undeclared function 'printg'; ISO C99 and later do not support implicit function declarations}} \
                                 expected-note{{did you mean 'printf'?}} \
                                 c2x-error {{use of undeclared identifier 'printg'; did you mean 'printf'?}}

  __builtin_is_les(1, 3); // expected-error{{use of unknown builtin '__builtin_is_les'}} \
                             c2x-error {{unknown type name '__builtin_is_les'; did you mean '__builtin_va_list'?}} \
                             c2x-error {{expected identifier or '('}} \
                             c2x-error {{expected ')'}} \
                             c2x-note {{to match this '('}}
}
Boolean _CFCalendarDecomposeAbsoluteTimeV(const char *componentDesc, int32_t **vector, int32_t count) { // expected-error {{conflicting types}}
 return 0;
}


// Test the typo-correction callback in Sema::ImplicitlyDefineFunction
extern int sformatf(char *str, __const char *__restrict __format, ...); // expected-note{{'sformatf' declared here}}
void test_implicit(void) {
  int formats = 0; // c2x-note {{'formats' declared here}}
  formatd("Hello, World!\n"); // expected-error{{call to undeclared function 'formatd'; ISO C99 and later do not support implicit function declarations}} \
                                 expected-note{{did you mean 'sformatf'?}} \
                                 c2x-error {{use of undeclared identifier 'formatd'; did you mean 'formats'?}} \
                                 c2x-error {{called object type 'int' is not a function or function pointer}}
}

void test_suggestion(void) {
  bark(); // expected-error {{call to undeclared function 'bark'; ISO C99 and later do not support implicit function declarations}} \
             c2x-error {{use of undeclared identifier 'bark'}}
  bork(); // expected-error {{call to undeclared function 'bork'; ISO C99 and later do not support implicit function declarations}} \
             c2x-error {{use of undeclared identifier 'bork'}}
}
