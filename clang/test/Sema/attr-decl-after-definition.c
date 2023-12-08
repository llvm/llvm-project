// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo(void);
void foo(void) __attribute__((unused));
void foo(void) __attribute__((unused));
void foo(void){} // expected-note {{previous definition is here}}
void foo(void) __attribute__((constructor)); // expected-warning {{must precede definition}}
void foo(void);

int bar;
extern int bar;
int bar;
int bar __attribute__((weak));
int bar __attribute__((used));
extern int bar __attribute__((weak));
int bar = 0; // expected-note {{previous definition is here}}
int bar __attribute__((weak)); // no warning as it matches the existing
                               // attribute.
int bar __attribute__((used,
                       visibility("hidden"))); // expected-warning {{must precede definition}}
int bar;

struct zed {  // expected-note {{previous definition is here}}
};
struct __attribute__((visibility("hidden"))) zed; // expected-warning {{must precede definition}}

struct __attribute__((visibility("hidden"))) zed2 {
};
struct __attribute__((visibility("hidden"))) zed2;

struct __attribute__((visibility("hidden"))) zed3 {  // expected-note {{previous definition is here}}
};
struct __attribute__((visibility("hidden"),
                     packed  // expected-warning {{must precede definition}}
                     )) zed3;

struct __attribute__((visibility("hidden"))) zed4 {  // expected-note {{previous attribute is here}}
};
struct __attribute__((visibility("default"))) zed4; // expected-error {{visibility does not match previous declaration}}
