// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin10 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -triple x86_64-apple-darwin10 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -triple x86_64-apple-darwin10 -std=c++98 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -triple x86_64-apple-darwin10 -std=c++11 %s

int f(int i) {
  switch (i) {
    case 2147483647 + 2:
#if (__cplusplus <= 199711L) // C or C++03 or earlier modes
    // expected-warning@-2 {{overflow in expression; result is -2'147'483'647 with type 'int'}}
#else
    // expected-error@-4 {{case value is not a constant expression}} \
    // expected-note@-4 {{value 2147483649 is outside the range of representable values of type 'int'}}
#endif
      return 1;
    case 9223372036854775807L * 4:
#if (__cplusplus <= 199711L)
    // expected-warning@-2 {{overflow in expression; result is -4 with type 'long'}}
#else
    // expected-error@-4 {{case value is not a constant expression}} \
    // expected-note@-4 {{value 36893488147419103228 is outside the range of representable values of type 'long'}}
#endif
      return 2;
    case (123456 *789012) + 1:
#if (__cplusplus <= 199711L)
    // expected-warning@-2 {{overflow in expression; result is -1'375'982'336 with type 'int'}}
#else
    // expected-error@-4 {{case value is not a constant expression}} \
    // expected-note@-4 {{value 97408265472 is outside the range of representable values of type 'int'}}
#endif
      return 3;
    case (2147483647*4)/4:
#if (__cplusplus <= 199711L)
    // expected-warning@-2 {{overflow in expression; result is -4 with type 'int'}}
#else
    // expected-error@-4 {{case value is not a constant expression}} \
    // expected-note@-4 {{value 8589934588 is outside the range of representable values of type 'int'}}
#endif
    case (2147483647*4)%4:
#if (__cplusplus <= 199711L)
    // expected-warning@-2 {{overflow in expression; result is -4 with type 'int'}}
#else
    // expected-error@-4 {{case value is not a constant expression}} \
    // expected-note@-4 {{value 8589934588 is outside the range of representable values of type 'int'}}
#endif
      return 4;
    case 2147483647:
      return 0;
  }
  return (i, 65537) * 65537; // expected-warning {{overflow in expression; result is 131'073 with type 'int'}} \
			     // expected-warning {{left operand of comma operator has no effect}}
}

unsigned long long l = 65536 * 65536; // expected-warning {{overflow in expression; result is 0 with type 'int'}}
unsigned long long l2 = 65536 * (unsigned)65536;
unsigned long long l3 = 65536 * 65536ULL;
