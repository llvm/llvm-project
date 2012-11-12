// RUN: %clang_cc1 -w -fdump-record-layouts %s 2> %t.layouts
// RUN: %clang_cc1 -w -fdump-record-layouts-simple %s > %t.before 2>&1
// RUN: %clang_cc1 -w -DPACKED= -DALIGNED16= -fdump-record-layouts-simple -foverride-record-layout=%t.layouts %s > %t.after 2>&1
// RUN: diff %t.before %t.after
// RUN: FileCheck %s < %t.after

// If not explicitly disabled, set PACKED to the packed attribute.
#ifndef PACKED
#  define PACKED __attribute__((packed))
#endif

// If not explicitly disabled, set ALIGNED16 to 16-byte alignment.
#ifndef ALIGNED16
#  define ALIGNED16 __attribute__((aligned(16)))
#endif

// CHECK: Type: struct X0
struct X0 { 
  int x[6] PACKED; 
};

// CHECK: Type: struct X1
struct X1 { 
  char x[13]; 
  struct X0 y; 
} PACKED;

// CHECK: Type: struct X2
struct PACKED X2 {
  short x;
  int y;
};

// CHECK: Type: struct X3
struct X3 {
  short x PACKED;
  int y;
};

#pragma pack(push,2)
// CHECK: Type: struct X4
struct X4 {
  int x;
  int y;
};
#pragma pack(pop)

// CHECK: Type: struct X5
struct PACKED X5 { double a[19];  signed char b; };

// CHECK: Type: struct X6
struct PACKED X6 { long double a; char b; };

// CHECK: Type: struct X7
struct X7 {
        unsigned x;
        unsigned char y;
} PACKED;

// CHECK: Type: union X8
union X8 {
  struct X7 x;
  unsigned y;
} PACKED;

// CHECK: Type: struct X9
struct X9 {
  unsigned int x[2] PACKED;
  unsigned int y;
  unsigned int z PACKED;
};

// CHECK: Type: struct X10
struct X10 {
  unsigned int x[2] PACKED;
  unsigned int y PACKED;
  unsigned int z PACKED;
};

// CHECK: Type: struct X11
struct PACKED X11 {
  unsigned int x[2];
  unsigned int y;
  unsigned int z;
};

// CHECK: Type: struct X12
struct PACKED X12 {
  int x : 24;
};

// CHECK: Type: struct X13
struct PACKED X13 {
  signed x : 10;
  signed y : 10;
};

// CHECK: Type: union X14
union PACKED X14 {
  unsigned long long x : 3;
};

// CHECK: Type: struct X15
struct X15 {
  unsigned x : 16;
  unsigned y : 28 PACKED;
};

// CHECK: Type: struct X16
struct ALIGNED16 X16 {
  int a, b, c;
  int x : 5;
  int y : 29;
};

void use_structs() {
  struct X0 x0;
  x0.x[5] = sizeof(struct X0);

  struct X1 x1;
  x1.x[5] = sizeof(struct X1);

  struct X2 x2;
  x2.y = sizeof(struct X2);

  struct X3 x3;
  x3.y = sizeof(struct X3);

  struct X4 x4;
  x4.y = sizeof(struct X4);

  struct X5 x5;
  x5.b = sizeof(struct X5);

  struct X6 x6;
  x6.b = sizeof(struct X6);

  struct X7 x7;
  typedef int X7array[sizeof(struct X7)];
  x7.x = sizeof(struct X7);
  x7.y = x7.x;

  union X8 x8;
  typedef int X8array[sizeof(union X8)];
  x8.y = sizeof(union X8);
  x8.x.x = x8.y;

  struct X9 x9;
  typedef int X9array[sizeof(struct X9)];
  x9.y = sizeof(struct X9);

  struct X10 x10;
  typedef int X10array[sizeof(struct X10)];
  x10.y = sizeof(struct X10);

  struct X11 x11;
  typedef int X11array[sizeof(struct X11)];
  x11.y = sizeof(struct X11);

  struct X12 x12;
  x12.x = sizeof(struct X12);

  struct X13 x13;
  x13.x = sizeof(struct X13);

  union X14 x14;
  x14.x = sizeof(union X14);

  struct X15 x15;
  x15.x = sizeof(struct X15);

  struct X16 x16;
  x16.x = sizeof(struct X16);
}
