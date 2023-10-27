// RUN: %clang_cc1 -x c -triple x86_64-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#define __tag5 __attribute__((btf_type_tag("tag5")))
#define __tag6 __attribute__((btf_type_tag("tag6")))

int __attribute__((btf_type_tag("tag1", "tag2"))) *invalid1; // expected-error {{'btf_type_tag' attribute takes one argument}}
int __attribute__((btf_type_tag(2))) *invalid2; // expected-error {{expected string literal as argument of 'btf_type_tag' attribute}}

int * __tag1 __tag2 * __tag3 __tag4 * __tag5 __tag6 *g;

typedef void __fn_t(int);
typedef __fn_t __tag1 __tag2 * __tag3 __tag4 *__fn2_t;
struct t {
  int __tag1 * __tag2 * __tag3 *a;
  int __tag1 __tag2 __tag3 *b;
  __fn2_t c;
  long d;
};
int __tag4 * __tag5 * __tag6 *foo1(struct t __tag1 * __tag2 * __tag3 *a1) {
  return (int __tag4 * __tag5 * __tag6 *)a1[0][0]->d;
}

// The btf_type_tag attribute will be ignored during _Generic type matching
int g1 = _Generic((int *)0, int __tag1 *: 0);
int g2 = _Generic((int __tag1 *)0, int *: 0);
int g3 = _Generic(0,
                  int __tag1 * : 0, // expected-note {{compatible type 'int  __attribute__((btf_type_tag("tag1")))*' (aka 'int *') specified here}}
                  int * : 0, // expected-error {{type 'int *' in generic association compatible with previously specified type 'int  __attribute__((btf_type_tag("tag1")))*' (aka 'int *')}}
                  default : 0);

// The btf_type_tag attribute will be ignored during overloadable type matching
void bar2(int __tag1 *a) __attribute__((overloadable)) { asm volatile (""); } // expected-note {{previous definition is here}}
void bar2(int *a) __attribute__((overloadable)) { asm volatile (""); } // expected-error {{redefinition of 'bar2'}}
void foo2(int __tag1 *a, int *b) {
  bar2(a);
  bar2(b);
}
