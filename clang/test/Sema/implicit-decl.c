// RUN: %clang_cc1 %s -verify -fsyntax-only -Werror=implicit-function-declaration

/// -Werror-implicit-function-declaration is a deprecated alias used by many projects.
// RUN: %clang_cc1 %s -verify -fsyntax-only -Werror-implicit-function-declaration

typedef int int32_t;
typedef unsigned char Boolean;

extern int printf(__const char *__restrict __format, ...); // expected-note{{'printf' declared here}}

void func(void) {
   int32_t *vector[16];
   const char compDesc[16 + 1];
   int32_t compCount = 0;
   if (_CFCalendarDecomposeAbsoluteTimeV(compDesc, vector, compCount)) { // expected-error {{implicit declaration of function '_CFCalendarDecomposeAbsoluteTimeV' is invalid in C99}} expected-note {{previous implicit declaration}}
   }

   printg("Hello, World!\n"); // expected-error{{implicit declaration of function 'printg' is invalid in C99}} \
                              // expected-note{{did you mean 'printf'?}}

  __builtin_is_les(1, 3); // expected-error{{use of unknown builtin '__builtin_is_les'}}
}
Boolean _CFCalendarDecomposeAbsoluteTimeV(const char *componentDesc, int32_t **vector, int32_t count) { // expected-error {{conflicting types}}
 return 0;
}


// Test the typo-correction callback in Sema::ImplicitlyDefineFunction
extern int sformatf(char *str, __const char *__restrict __format, ...); // expected-note{{'sformatf' declared here}}
void test_implicit(void) {
  int formats = 0;
  formatd("Hello, World!\n"); // expected-error{{implicit declaration of function 'formatd' is invalid in C99}} \
                              // expected-note{{did you mean 'sformatf'?}}
}
