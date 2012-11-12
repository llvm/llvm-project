// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR3459
struct bar {
  char n[1];
};

struct foo {
  char name[(int)&((struct bar *)0)->n];
  char name2[(int)&((struct bar *)0)->n - 1]; //expected-error{{'name2' declared as an array with a negative size}}
};

// PR3430
struct s {
  struct st {
    int v;
  } *ts;
};

struct st;

int foo() {
  struct st *f;
  return f->v + f[0].v;
}

// PR3642, PR3671
struct pppoe_tag {
 short tag_type;
 char tag_data[];
};
struct datatag {
  struct pppoe_tag hdr; //expected-warning{{field 'hdr' with variable sized type 'struct pppoe_tag' not at the end of a struct or class is a GNU extension}}
  char data;
};


// PR4092
struct s0 {
  char a;  // expected-note {{previous declaration is here}}
  char a;  // expected-error {{duplicate member 'a'}}
};

struct s0 f0(void) {}

// <rdar://problem/8177927> - This previously triggered an assertion failure.
struct x0 {
  unsigned int x1;
};

// rdar://problem/9150338
static struct test1 { // expected-warning {{'static' ignored on this declaration}}
  int x;
};
const struct test2 { // expected-warning {{'const' ignored on this declaration}}
  int x;
};
inline struct test3 { // expected-warning {{'inline' ignored on this declaration}}
  int x;
};
