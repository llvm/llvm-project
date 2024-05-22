// RUN: %clang_cc1 -x c -triple x86_64-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))
#define __tag3 __attribute__((btf_decl_tag("tag3")))

#define __tag_no_arg __attribute__((btf_decl_tag()))
#define __tag_2_arg __attribute__((btf_decl_tag("tag1", "tag2")))
#define __invalid __attribute__((btf_decl_tag(1)))

struct __tag1 __tag2 t1;
struct t1 {
  int a __tag1;
} __tag3;

struct __tag1 t2;
struct __tag2 __tag3 t2 {
  int a __tag1;
};

int g1 __tag1;
int g2 __tag_no_arg; // expected-error {{'btf_decl_tag' attribute takes one argument}}
int g3 __tag_2_arg; // expected-error {{'btf_decl_tag' attribute takes one argument}}
int i1 __invalid; // expected-error {{expected string literal as argument of 'btf_decl_tag' attribute}}

enum e1 {
  E1
} __tag1; // expected-error {{'btf_decl_tag' attribute only applies to variables, functions, structs, unions, classes, non-static data members, and typedefs}}

enum e2 {
  E2
} __tag_no_arg; // expected-error {{'btf_decl_tag' attribute only applies to variables, functions, structs, unions, classes, non-static data members, and typedefs}}

enum e3 {
  E3
} __tag_2_arg; // expected-error {{'btf_decl_tag' attribute only applies to variables, functions, structs, unions, classes, non-static data members, and typedefs}}

int __tag1 __tag2 foo(struct t1 *arg, struct t2 *arg2);
int __tag2 __tag3 foo(struct t1 *arg, struct t2 *arg2);
int __tag1 foo(struct t1 *arg __tag1, struct t2 *arg2) {
  return arg->a + arg2->a;
}

typedef unsigned * __u1 __tag1 __tag2;
typedef struct {
  int a;
} __t2 __tag1 __tag2;
int convert2(__t2 *arg) {
  return arg->a;
}
