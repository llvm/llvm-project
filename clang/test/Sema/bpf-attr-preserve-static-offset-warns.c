// RUN: %clang_cc1 -fsyntax-only -verify -triple bpf-pc-linux-gnu %s

#define __pso __attribute__((preserve_static_offset))

// These are correct usages.
struct foo { int a; } __pso;
union quux { int a; } __pso;
struct doug { int a; } __pso __attribute__((packed));

// Rest are incorrect usages.
typedef int bar __pso;    // expected-error{{attribute only applies to}}
struct goo {
  int a __pso;            // expected-error{{attribute only applies to}}
};
int g __pso;              // expected-error{{attribute only applies to}}
__pso void ffunc1(void);  // expected-error{{attribute only applies to}}
void ffunc2(int a __pso); // expected-error{{attribute only applies to}}
void ffunc3(void) {
  int a __pso;            // expected-error{{attribute only applies to}}
}

struct buz { int a; } __attribute__((preserve_static_offset("hello"))); // \
  expected-error{{attribute takes no arguments}}
