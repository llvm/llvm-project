// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x c %t/Inputs/first.h
// RUN: %clang_cc1 -fsyntax-only -x c %t/Inputs/second.h

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I%t/Inputs -verify %s

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST)
struct S1 {};
struct S1 s1a;
#elif defined(SECOND)
struct S1 {};
#else
struct S1 s1;
#endif

#if defined(FIRST)
struct S2 {
  int x;
  int y;
};
#elif defined(SECOND)
struct S2 {
  int y;
  int x;
};
#else
struct S2 s2;
// expected-error@first.h:* {{'S2' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'x'}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'y'}}
#endif

#if defined(FIRST)
struct S3 {
  double x;
};
#elif defined(SECOND)
struct S3 {
  int x;
};
#else
struct S3 s3;
// expected-error@second.h:* {{'S3::x' from module 'SecondModule' is not present in definition of 'struct S3' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
typedef int A;
struct S4 {
  A x;
};

struct S5 {
  A x;
};
#elif defined(SECOND)
typedef int B;
struct S4 {
  B x;
};

struct S5 {
  int x;
};
#else
struct S4 s4;
// expected-error@first.h:* {{'S4' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'x' with type 'A' (aka 'int')}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'x' with type 'B' (aka 'int')}}

struct S5 s5;
// expected-error@first.h:* {{'S5' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'x' with type 'A' (aka 'int')}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'x' with type 'int'}}
#endif

#if defined(FIRST)
struct S6 {
  unsigned x;
};
#elif defined(SECOND)
struct S6 {
  unsigned x : 1;
};
#else
struct S6 s6;
// expected-error@first.h:* {{'S6' has different definitions in different modules; first difference is definition in module 'FirstModule' found non-bitfield 'x'}}
// expected-note@second.h:* {{but in 'SecondModule' found bitfield 'x'}}
#endif

#if defined(FIRST)
struct S7 {
  unsigned x : 2;
};
#elif defined(SECOND)
struct S7 {
  unsigned x : 1;
};
#else
struct S7 s7;
// expected-error@first.h:* {{'S7' has different definitions in different modules; first difference is definition in module 'FirstModule' found bitfield 'x' with one width expression}}
// expected-note@second.h:* {{but in 'SecondModule' found bitfield 'x' with different width expression}}
#endif

#if defined(FIRST)
struct S8 {
  unsigned x : 2;
};
#elif defined(SECOND)
struct S8 {
  unsigned x : 1 + 1;
};
#else
struct S8 s8;
// expected-error@first.h:* {{'S8' has different definitions in different modules; first difference is definition in module 'FirstModule' found bitfield 'x' with one width expression}}
// expected-note@second.h:* {{but in 'SecondModule' found bitfield 'x' with different width expression}}
#endif

#if defined(FIRST)
struct S12 {
  unsigned x[5];
};
#elif defined(SECOND)
struct S12 {
  unsigned x[7];
};
#else
struct S12 s12;
// expected-error@second.h:* {{'S12::x' from module 'SecondModule' is not present in definition of 'struct S12' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct S13 {
  unsigned x[7];
};
#elif defined(SECOND)
struct S13 {
  double x[7];
};
#else
struct S13 s13;
// expected-error@second.h:* {{'S13::x' from module 'SecondModule' is not present in definition of 'struct S13' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
struct B1 {};
struct SS1 {
  struct B1 x;
};
#elif defined(SECOND)
struct A1 {};
struct SS1 {
  struct A1 x;
};
#else
struct SS1 ss1;
// expected-error@second.h:* {{'SS1::x' from module 'SecondModule' is not present in definition of 'struct SS1' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif

#if defined(FIRST)
enum E1 { x42 };
struct SE1 {
  enum E1 x;
};
#elif defined(SECOND)
enum E2 { x42 };
struct SE1 {
  enum E2 x;
};
#else
struct SE1 se1;
// expected-error@second.h:* {{'SE1::x' from module 'SecondModule' is not present in definition of 'struct SE1' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'x' does not match}}
#endif

// struct with forward declaration
#if defined(FIRST)
struct P {};
struct S {
  struct P *ptr;
};
#elif defined(SECOND)
struct S {
  struct P *ptr;
};
#else
struct S s;
#endif

// struct with forward declaration and no definition
#if defined(FIRST)
struct PA;
struct SA {
  struct PA *ptr;
};
#elif defined(SECOND)
struct SA {
  struct PA *ptr;
};
#else
struct SA sa;
#endif

// struct with multiple typedefs
#if defined(FIRST)
typedef int BB1;
typedef BB1 AA1;
struct TS1 {
  AA1 x;
};
#elif defined(SECOND)
typedef int AA1;
struct TS1 {
  AA1 x;
};
#else
struct TS1 ts1;
#endif

#if defined(FIRST)
struct T2 {
  int x;
};
typedef struct T2 B2;
typedef struct B2 A2;
struct TS2 {
  struct T2 x;
};
#elif defined(SECOND)
struct T2 {
  int x;
};
typedef struct T2 A2;
struct TS2 {
  struct T2 x;
};
#else
struct TS2 ts2;
#endif

#if defined(FIRST)
struct T3;
struct TS3 {
  struct T3 *t;
};
#elif defined(SECOND)
typedef struct T3 {
} T3;
struct TS3 {
  struct T3 *t;
};
#else
struct TS3 ts3;
#endif

#if defined(FIRST)
struct AU {
  union {
    int a;
  };
};
#elif defined(SECOND)
struct AU {
  union {
    char a;
  };
};
#else
struct AU au;
// expected-error@first.h:* {{'AU' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'a' with type 'int'}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'a' with type 'char'}}
#endif

#if defined(FIRST)
struct AUS {
  union {
    int a;
  };
  struct {
    char b;
  };
};
#elif defined(SECOND)
struct AUS {
  union {
    int a;
  };
  struct {
    int b;
  };
};
#else
struct AUS aus;
// expected-error@first.h:* {{'AUS' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'b' with type 'char'}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'b' with type 'int'}}
#endif

#if defined(FIRST)
struct AUS1 {
  union {
    int x;
    union {
      int y;
      struct {
        int z;
      };
    };
  };
};
#elif defined(SECOND)
struct AUS1 {
  union {
    int x;
    union {
      int y;
      struct {
        char z;
      };
    };
  };
};
#else
struct AUS1 aus1;
// expected-error@first.h:* {{'AUS1' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'z' with type 'int'}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'z' with type 'char'}}
#endif

#if defined(FIRST)
union U {
  int a;
  char b;
};
#elif defined(SECOND)
union U {
  int a;
  float b;
};
#else
union U u;
// expected-error@second.h:* {{'U::b' from module 'SecondModule' is not present in definition of 'union U' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'b' does not match}}
#endif

#if defined(FIRST)
struct TSS1 {
  int tss1;
};
#elif defined(SECOND)
typedef struct TSS1 TSS1;
#else
struct TSS1 *tss1;
#endif

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
