// RUN: %clang_cc1 %s -verify -fsyntax-only

struct s { int a; } __attribute__((deprecated)) x;  // expected-warning {{'s' is deprecated}} expected-note 2 {{'s' declared here}}

typeof(x) y;  // expected-warning {{'s' is deprecated}}

union un{ int a; } __attribute__((deprecated)) u;  // expected-warning {{'un' is deprecated}} expected-note 2 {{'un' declared here}}

typeof(     u) z; // expected-warning {{'un' is deprecated}}

enum E{ one} __attribute__((deprecated))  e; // expected-warning {{'E' is deprecated}} expected-note 2 {{'E' declared here}}

typeof( e) w; // expected-warning {{'E' is deprecated}}

struct foo { int x; } __attribute__((deprecated)); // expected-note {{'foo' declared here}}
typedef struct foo bar __attribute__((deprecated)); // expected-note {{'bar' declared here}}
bar x1;	// expected-warning {{'bar' is deprecated}}

int main() { typeof(x1) y; }	// expected-warning {{'foo' is deprecated}}

struct gorf { int x; };
typedef struct gorf T __attribute__((deprecated));  // expected-note {{'T' declared here}}
T t;	// expected-warning {{'T' is deprecated}}
void wee() { typeof(t) y; }


