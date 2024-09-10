// RUN: %clang_cc1 %s -verify=stock,c -fsyntax-only
// RUN: %clang_cc1 %s -verify=stock,cpp -fsyntax-only -x c++
// RUN: %clang_cc1 %s -verify=stock,cpp -fsyntax-only -fms-compatibility -x c++
// RUN: %clang_cc1 %s -verify=stock,c,gnu -fsyntax-only -Wgnu-flexible-array-union-member -Wgnu-empty-struct
// RUN: %clang_cc1 %s -verify=stock,c,microsoft -fsyntax-only -fms-compatibility -Wmicrosoft

// The test checks that an attempt to initialize union with flexible array
// member with an initializer list doesn't crash clang.


union { char x[]; } r = {0}; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                                microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
                              */
struct _name1 {
  int a;
  union {
    int b;
    char x[]; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
  };
} name1 = {
  10,
  42,        /* initializes "b" */
};

struct _name1i {
  int a;
  union {
    int b;
    char x[]; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
  };
} name1i = {
  .a = 10,
  .b = 42,
};

/* Initialization of flexible array in a union is never allowed. */
struct _name2 {
  int a;
  union {
    int b;
    char x[]; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
  };
} name2 = {
  12,
  13,
  { 'c' },   /* c-warning {{excess elements in struct initializer}}
                cpp-error {{excess elements in struct initializer}}
              */
};

/* Initialization of flexible array in a union is never allowed. */
struct _name2i {
  int a;
  union {
    int b;
    char x[]; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
                 stock-note {{initialized flexible array member 'x' is here}}
               */
  };
} name2i = {
  .a = 12,
  .b = 13,      /* stock-note {{previous initialization is here}} */
  .x = { 'c' }, /* stock-error {{initialization of flexible array member is not allowed}}
                   c-warning {{initializer overrides prior initialization of this subobject}}
                   cpp-error {{initializer partially overrides prior initialization of this subobject}}
                 */
};

/* Flexible array initialization always allowed when not in a union,
   and when struct has another member.
 */
struct _okay {
  int a;
  char x[];
} okay = {
  22,
  { 'x', 'y', 'z' },
};

struct _okayi {
  int a;
  char x[];
} okayi = {
  .a = 22,
  .x = { 'x', 'y', 'z' },
};

struct _okay0 {
  int a;
  char x[];
} okay0 = { };

struct _flex_extension {
  char x[]; /* gnu-warning {{flexible array member 'x' in otherwise empty struct is a GNU extension}}
               microsoft-warning {{flexible array member 'x' in otherwise empty struct is a Microsoft extension}}
             */
} flex_extension = {
  { 'x', 'y', 'z' },
};

struct _flex_extensioni {
  char x[]; /* gnu-warning {{flexible array member 'x' in otherwise empty struct is a GNU extension}}
               microsoft-warning {{flexible array member 'x' in otherwise empty struct is a Microsoft extension}}
             */
} flex_extensioni = {
  .x = { 'x', 'y', 'z' },
};

struct already_hidden {
  int a;
  union {
    int b;
    struct {
      struct { } __empty;  // gnu-warning {{empty struct is a GNU extension}}
      char x[];
    };
  };
};

struct still_zero_sized {
  struct { } __unused;  // gnu-warning {{empty struct is a GNU extension}}
  int x[];
};

struct warn1 {
  int a;
  union {
    int b;
    char x[]; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
  };
};

struct warn2 {
  int x[];  /* gnu-warning {{flexible array member 'x' in otherwise empty struct is a GNU extension}}
               microsoft-warning {{flexible array member 'x' in otherwise empty struct is a Microsoft extension}}
             */
};

union warn3 {
  short x[];  /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
};

struct quiet1 {
  int a;
  short x[];
};

struct _not_at_end {
  union { short x[]; }; /* stock-warning-re {{field '' with variable sized type '{{.*}}' not at the end of a struct or class is a GNU extension}}
                           gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                           microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
                         */
  int y;
} not_at_end = {{}, 3};

struct _not_at_end_s {
  struct { int a; short x[]; }; /* stock-warning-re {{field '' with variable sized type '{{.*}}' not at the end of a struct or class is a GNU extension}} */
  int y;
} not_at_end_s = {{}, 3};

struct {
  int a;
  union {      /* stock-warning-re {{field '' with variable sized type '{{.*}}' not at the end of a struct or class is a GNU extension}} */
    short x[]; /* stock-note {{initialized flexible array member 'x' is here}}
                  gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                  microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
                */
    int b;
  };
  int c;
  int d;
} i_f = { 4,
         {5},  /* stock-error {{initialization of flexible array member is not allowed}} */
         {},
          6};

// expected-no-diagnostics
