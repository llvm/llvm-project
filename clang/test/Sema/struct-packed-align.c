// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

// Packed structs.
struct s {
    char a;
    int b  __attribute__((packed));
    char c;
    int d;
};

extern int a1[sizeof(struct s) == 12 ? 1 : -1];
extern int a2[__alignof(struct s) == 4 ? 1 : -1];

struct __attribute__((packed)) packed_s {
    char a;
    int b  __attribute__((packed));
    char c;
    int d;
};

extern int b1[sizeof(struct packed_s) == 10 ? 1 : -1];
extern int b2[__alignof(struct packed_s) == 1 ? 1 : -1];

struct fas {
    char a;
    int b[];
};

extern int c1[sizeof(struct fas) == 4 ? 1 : -1];
extern int c2[__alignof(struct fas) == 4 ? 1 : -1];

struct __attribute__((packed)) packed_fas {
    char a;
    int b[];
};

extern int d1[sizeof(struct packed_fas) == 1 ? 1 : -1];
extern int d2[__alignof(struct packed_fas) == 1 ? 1 : -1];

struct packed_after_fas {
    char a;
    int b[];
} __attribute__((packed));

extern int d1_2[sizeof(struct packed_after_fas) == 1 ? 1 : -1];
extern int d2_2[__alignof(struct packed_after_fas) == 1 ? 1 : -1];

// Alignment

struct __attribute__((aligned(8))) as1 {
    char c;
};

extern int e1[sizeof(struct as1) == 8 ? 1 : -1];
extern int e2[__alignof(struct as1) == 8 ? 1 : -1];

// FIXME: Will need to force arch once max usable alignment isn't hard
// coded.
struct __attribute__((aligned)) as1_2 {
    char c;
};
extern int e1_2[sizeof(struct as1_2) == 16 ? 1 : -1];
extern int e2_2[__alignof(struct as1_2) == 16 ? 1 : -1];

struct as2 {
    char c;
    int __attribute__((aligned(8))) a;
};

extern int f1[sizeof(struct as2) == 16 ? 1 : -1];
extern int f2[__alignof(struct as2) == 8 ? 1 : -1];

struct __attribute__((packed)) as3 {
    char c;
    int a;
    int __attribute__((aligned(8))) b;
};

extern int g1[sizeof(struct as3) == 16 ? 1 : -1];
extern int g2[__alignof(struct as3) == 8 ? 1 : -1];


// rdar://5921025
struct packedtest {
  int ted_likes_cheese;
  void *args[] __attribute__((packed));
};

// Packed union
union __attribute__((packed)) au4 {char c; int x;};
extern int h1[sizeof(union au4) == 4 ? 1 : -1];
extern int h2[__alignof(union au4) == 1 ? 1 : -1];

// Aligned union
union au5 {__attribute__((aligned(4))) char c;};
extern int h1[sizeof(union au5) == 4 ? 1 : -1];
extern int h2[__alignof(union au5) == 4 ? 1 : -1];

// Alignment+packed
struct as6 {char c; __attribute__((packed, aligned(2))) int x;};
extern int i1[sizeof(struct as6) == 6 ? 1 : -1];
extern int i2[__alignof(struct as6) == 2 ? 1 : -1];

union au6 {char c; __attribute__((packed, aligned(2))) int x;};
extern int k1[sizeof(union au6) == 4 ? 1 : -1];
extern int k2[__alignof(union au6) == 2 ? 1 : -1];

// Check postfix attributes
union au7 {char c; int x;} __attribute__((packed));
extern int l1[sizeof(union au7) == 4 ? 1 : -1];
extern int l2[__alignof(union au7) == 1 ? 1 : -1];

struct packed_fas2 {
    char a;
    int b[];
} __attribute__((packed));

extern int m1[sizeof(struct packed_fas2) == 1 ? 1 : -1];
extern int m2[__alignof(struct packed_fas2) == 1 ? 1 : -1];

// Attribute aligned can round down typedefs.  PR9253
typedef long long  __attribute__((aligned(1))) nt;

struct nS {
  char buf_nr;
  nt start_lba;
};

extern int n1[sizeof(struct nS) == 9 ? 1 : -1];
extern int n2[__alignof(struct nS) == 1 ? 1 : -1];




