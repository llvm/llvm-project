// RUN: %clang_cc1 -fsyntax-only -Wuninitialized-explicit-init -verify %s

struct ops {
    int (*show)(void) __attribute__((require_explicit_initialization)); // #SHOW
    int flags;
};

struct plain {
    int a;
};

struct plain no_attribute; // no warning: no field requires explicit initialization

struct ops never_initialized; // expected-warning {{field in 'ops' requires explicit initialization but is not explicitly initialized}} expected-note@#SHOW {{'show' declared here}}
static struct ops never_initialized_static; // expected-warning {{field in 'ops' requires explicit initialization but is not explicitly initialized}} expected-note@#SHOW {{'show' declared here}}

struct ops repeated;
struct ops repeated; // expected-warning {{field in 'ops' requires explicit initialization but is not explicitly initialized}} expected-note@#SHOW {{'show' declared here}}

struct ops initialized_later;
struct ops initialized_later = { .flags=1}; // expected-warning {{field 'show' requires explicit initialization but is not explicitly initialized}} expected-note@#SHOW {{'show' declared here}}

struct ops initialized_ok;
struct ops initialized_ok = { .show=0};

extern struct ops ext;

struct late g_late; // expected-warning {{field in 'late' requires explicit initialization but is not explicitly initialized}} expected-note@#LATE_X {{'x' declared here}}
struct late {
    int x __attribute__((require_explicit_initialization)); // #LATE_X
};

void f(void){
    static struct ops local_static; // expected-warning {{field in 'ops' requires explicit initialization but is not explicitly initialized}} expected-note@#SHOW {{'show' declared here}}
    (void)&local_static;
}
