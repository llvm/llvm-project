// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

#define __reloc__ __attribute__((preserve_access_index))
#define __err_reloc__ __attribute__((preserve_access_index(0)))

struct t1 {
  int a;
  int b[4];
  int c:1;
} __reloc__;

union t2 {
  int a;
  int b[4];
  int c:1;
} __reloc__;

struct t3 {
  int a;
} __err_reloc__; // expected-error {{'preserve_access_index' attribute takes no arguments}}

struct t4 {
  union {
    int a;
    char b[5];
  };
  struct {
    int c:1;
  } __reloc__;
  int d;
} __reloc__;

struct __reloc__ p;
struct __reloc__ q;
struct p {
  int a;
};

int a __reloc__; // expected-error {{'preserve_access_index' attribute only applies to structs, unions, and classes}}
struct s *p __reloc__; // expected-error {{'preserve_access_index' attribute only applies to structs, unions, and classes}}

void invalid1(const int __reloc__ *arg) {} // expected-error {{'preserve_access_index' attribute only applies to structs, unions, and classes}}
void invalid2() { const int __reloc__ *arg; } // expected-error {{'preserve_access_index' attribute only applies to structs, unions, and classes}}
int valid3(struct t4 *arg) { return arg->a + arg->b[3] + arg->c + arg->d; }
int valid4(void *arg) {
  struct local_t { int a; int b; } __reloc__;
  return ((struct local_t *)arg)->b;
}
