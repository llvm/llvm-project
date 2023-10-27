// RUN: %clang_cc1 -Wno-pointer-to-int-cast -fsyntax-only -verify %s
// PR3459
struct bar {
  char n[1];
};

struct foo {
  char name[(int)&((struct bar *)0)->n]; // expected-warning {{folded to constant}}
  char name2[(int)&((struct bar *)0)->n - 1]; // expected-error {{array size is negative}}
};

// PR3430
struct s {
  struct st {
    int v;
  } *ts;
};

struct st;

int foo(void) {
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

// This previously triggered an assertion failure.
struct x0 {
  unsigned int x1;
};

static struct test1 { // expected-warning {{'static' ignored on this declaration}}
  int x;
};
const struct test2 { // expected-warning {{'const' ignored on this declaration}}
  int x;
};
inline struct test3 { // expected-error {{'inline' can only appear on functions}}
  int x;
};

struct hiding_1 {};
struct hiding_2 {};
void test_hiding(void) {
  struct hiding_1 *hiding_1(void);
  extern struct hiding_2 *hiding_2;
  struct hiding_1 *p = hiding_1();
  struct hiding_2 *q = hiding_2;
}

struct PreserveAttributes {};
typedef struct __attribute__((noreturn)) PreserveAttributes PreserveAttributes_t; // expected-warning {{'noreturn' attribute only applies to functions and methods}}

// PR46255
struct FlexibleArrayMem {
  int a;
  int b[];
};

struct FollowedByNamed {
  struct FlexibleArrayMem a; // expected-warning {{field 'a' with variable sized type 'struct FlexibleArrayMem' not at the end of a struct or class is a GNU extension}}
  int i;
};

struct FollowedByUnNamed {
  struct FlexibleArrayMem a; // expected-warning {{field 'a' with variable sized type 'struct FlexibleArrayMem' not at the end of a struct or class is a GNU extension}}
  struct {
    int i;
  };
};

struct InAnonymous {
  struct { // expected-warning-re {{field '' with variable sized type 'struct InAnonymous::(anonymous at {{.+}})' not at the end of a struct or class is a GNU extension}}

    struct FlexibleArrayMem a;
  };
  int i;
};
struct InAnonymousFollowedByAnon {
  struct { // expected-warning-re {{field '' with variable sized type 'struct InAnonymousFollowedByAnon::(anonymous at {{.+}})' not at the end of a struct or class is a GNU extension}}

    struct FlexibleArrayMem a;
  };
  struct {
    int i;
  };
};

// This is the behavior in C++ as well, so making sure we reproduce it here.
struct InAnonymousFollowedByEmpty {
  struct FlexibleArrayMem a; // expected-warning {{field 'a' with variable sized type 'struct FlexibleArrayMem' not at the end of a struct or class is a GNU extension}}
  struct {};
};
