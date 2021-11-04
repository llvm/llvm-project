// RUN: %clang_cc1 -x c -triple x86_64-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#define __tag5 __attribute__((btf_type_tag("tag5")))
#define __tag6 __attribute__((btf_type_tag("tag6")))

int __attribute__((btf_type_tag("tag1", "tag2"))) *invalid1; // expected-error {{'btf_type_tag' attribute takes one argument}}
int __attribute__((btf_type_tag(2))) *invalid2; // expected-error {{'btf_type_tag' attribute requires a string}}

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
