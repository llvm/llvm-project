// RUN: %clang_cc1 %s -verify -fsyntax-only

typedef int INT1 __attribute__((deprecated("Please avoid INT1"))); // expected-note 3 {{'INT1' has been explicitly marked deprecated here}}

typedef INT1 INT2 __attribute__ ((__deprecated__("Please avoid INT2")));

typedef INT1 INT1a; // expected-warning {{'INT1' is deprecated: Please avoid INT1}}

typedef INT1 INT1b __attribute__ ((deprecated("Please avoid INT1b")));

INT1 should_be_unavailable; // expected-warning {{'INT1' is deprecated: Please avoid INT1}}
INT1a should_not_be_deprecated;

INT1 f1(void) __attribute__ ((deprecated("Please avoid f1"))); // expected-note {{'f1' has been explicitly marked deprecated here}}
INT1 f2(void); // expected-warning {{'INT1' is deprecated: Please avoid INT1}}

typedef enum {red, green, blue} Color __attribute__((deprecated("Please avoid Color"))); // expected-note {{'Color' has been explicitly marked deprecated here}}
 

Color c1; // expected-warning {{'Color' is deprecated: Please avoid Color}}

int g1;
int g2 __attribute__ ((deprecated("Please avoid g2"))); // expected-note {{'g2' has been explicitly marked deprecated here}}

int func1(void)
{
   int (*pf)(void) = f1; // expected-warning {{'f1' is deprecated: Please avoid f1}}
   int i = f2();
   return g1 + g2; // expected-warning {{'g2' is deprecated: Please avoid g2}}
}
