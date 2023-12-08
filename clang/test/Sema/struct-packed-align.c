// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple=x86_64-windows-coff -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple=x86_64-scei-ps4 -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple=x86_64-sie-ps5 -verify

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

struct __attribute__((aligned)) as1_2 {
    char c;
};
#if ( defined(__s390x__) || ( defined (__ARM_32BIT_STATE) && ! defined(__ANDROID__) ) )
extern int e1_2[sizeof(struct as1_2) == 8 ? 1 : -1];
extern int e2_2[__alignof(struct as1_2) == 8 ? 1 : -1];
#else
extern int e1_2[sizeof(struct as1_2) == 16 ? 1 : -1];
extern int e2_2[__alignof(struct as1_2) == 16 ? 1 : -1];
#endif

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

#if defined(_WIN32) && !defined(__declspec) // _MSC_VER is unavailable in cc1.
// Alignment doesn't affect packing in MS mode.
extern int n1[sizeof(struct nS) == 16 ? 1 : -1];
extern int n2[__alignof(struct nS) == 8 ? 1 : -1];
#else
extern int n1[sizeof(struct nS) == 9 ? 1 : -1];
extern int n2[__alignof(struct nS) == 1 ? 1 : -1];
#endif

// Packed attribute shouldn't be ignored for bit-field of char types.
// Note from GCC reference manual: The 4.1, 4.2 and 4.3 series of GCC ignore
// the packed attribute on bit-fields of type char. This has been fixed in
// GCC 4.4 but the change can lead to differences in the structure layout.
// See the documentation of -Wpacked-bitfield-compat for more information.
struct packed_chars {
  char a : 8, b : 8, c : 8, d : 4;
#ifdef __SCE__
  // Test for pre-r254596 clang behavior on the PS4/PS5 targets, which must
  // maintain ABI backwards compatibility.
  char e : 8 __attribute__((packed));
  // expected-warning@-1 {{'packed' attribute ignored for field of type 'char'}}
#else
  char e : 8 __attribute__((packed));
  // expected-warning@-1 {{'packed' attribute was ignored on bit-fields with single-byte alignment in older versions of GCC and Clang}}
#endif
  char f : 4, g : 8, h : 8, i : 8;
};

#if (defined(_WIN32) || defined(__SCE__)) && !defined(__declspec) // _MSC_VER is unavailable in cc1.
// On Windows clang uses MSVC compatible layout in this case.
//
// Additionally, test for pre-r254596 clang behavior on the PS4/PS5 targets.
// They must maintain ABI backwards compatibility.
extern int o1[sizeof(struct packed_chars) == 9 ? 1 : -1];
extern int o2[__alignof(struct packed_chars) == 1 ? 1 : -1];
#elif defined(_AIX)
// On AIX, char bitfields have the same alignment as unsigned int.
extern int o1[sizeof(struct packed_chars) == 8 ? 1 : -1];
extern int o2[__alignof(struct packed_chars) == 4 ? 1 : -1];
#else
extern int o1[sizeof(struct packed_chars) == 8 ? 1 : -1];
extern int o2[__alignof(struct packed_chars) == 1 ? 1 : -1];
#endif
