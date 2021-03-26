// RUN: %clang_cc1 -emit-llvm-only -triple s390x-none-zos -fdump-record-layouts %s | FileCheck %s

struct s0 {
  short a:3;
  long b:5;
  int c:1;
  long d:10;
  char e:5;
} S0;
// CHECK:              0 | struct s0
// CHECK-NEXT:     0:0-2 |   short a
// CHECK-NEXT:     0:3-7 |   long b
// CHECK-NEXT:     1:0-0 |   int c
// CHECK-NEXT:    1:1-10 |   long d
// CHECK-NEXT:     2:3-7 |   char e
// CHECK-NEXT:           | [sizeof=3, align=1]

struct s1 {
  char a:7;
  long b:27;
  int c:2;
} S1;
// CHECK:              0 | struct s1
// CHECK-NEXT:     0:0-6 |   char a
// CHECK-NEXT:    0:7-33 |   long b
// CHECK-NEXT:     4:2-3 |   int c
// CHECK-NEXT:           | [sizeof=5, align=1]

struct s2 {
  char a:7;
  char  :0;
  short :0;
  short :0;
} S2;
// CHECK:              0 | struct s2
// CHECK-NEXT:     0:0-6 |   char a
// CHECK-NEXT:       4:- |   char
// CHECK-NEXT:       4:- |   short
// CHECK-NEXT:       4:- |   short
// CHECK-NEXT:           | [sizeof=4, align=4]

struct s3 {
  int a;
  int b:16;
  char  :0;
  char c:1;
} S3;
// CHECK:              0 | struct s3
// CHECK-NEXT:         0 |   int a
// CHECK-NEXT:    4:0-15 |   int b
// CHECK-NEXT:       8:- |   char
// CHECK-NEXT:     8:0-0 |   char c
// CHECK-NEXT:           | [sizeof=12, align=4]

struct s4 {
 unsigned int __attribute__((aligned(32))) a;
} S4;
// CHECK:              0 | struct s4
// CHECK-NEXT:         0 |   unsigned int a
// CHECK-NEXT:           | [sizeof=32, align=32]

struct s5 {
  char a;
  int  b:19 __attribute__((aligned(4)));
  int  c:22 __attribute__((aligned(8)));
  int  :0;
  int  d:10;
} S5;
// CHECK:              0 | struct s5
// CHECK-NEXT:         0 |   char a
// CHECK-NEXT:    4:0-18 |   int b
// CHECK-NEXT:    8:0-21 |   int c
// CHECK-NEXT:      12:- |   int
// CHECK-NEXT:    12:0-9 |   int d
// CHECK-NEXT:           | [sizeof=16, align=8]

struct s6 {
  char * a;
  char * b[];
} S6;
// CHECK:              0 | struct s6
// CHECK-NEXT:         0 |   char * a
// CHECK-NEXT:         8 |   char *[] b
// CHECK-NEXT:           | [sizeof=8, align=8]

struct s7 {
  long  :0;
  short a;
} S7;
// CHECK:              0 | struct s7
// CHECK-NEXT:       0:- |   long
// CHECK-NEXT:         0 |   short a
// CHECK-NEXT:           | [sizeof=2, align=2]

#pragma pack(2)
struct s8 {
  unsigned long       :0;
  long long           a;
} S8;
#pragma pack()
// CHECK:              0 | struct s8
// CHECK-NEXT:       0:- |   unsigned long
// CHECK-NEXT:         0 |   long long a
// CHECK-NEXT:           | [sizeof=8, align=2]

struct s9 {
  unsigned int   :0;
  unsigned short :0;
} S9;
// CHECK:              0 | struct s9
// CHECK-NEXT:       0:- |   unsigned int
// CHECK-NEXT:       0:- |   unsigned short
// CHECK-NEXT:           | [sizeof=0, align=1]

struct s10 {
 unsigned int __attribute__((aligned)) a;
} S10;
// CHECK:              0 | struct s10
// CHECK-NEXT:         0 |   unsigned int a
// CHECK-NEXT:           | [sizeof=16, align=16]

struct s11 {
  char a;
  long :0;
  char b;
} S11;
// CHECK:              0 | struct s11
// CHECK-NEXT:         0 |   char a
// CHECK-NEXT:       8:- |   long
// CHECK-NEXT:         8 |   char b
// CHECK-NEXT:           | [sizeof=16, align=8]

union u0 {
  unsigned short     d1 __attribute__((packed));
  int                d2:10;
  long               d3;
} U0 __attribute__((aligned(8)));
// CHECK:              0 | union u0
// CHECK-NEXT:         0 |   unsigned short d1
// CHECK-NEXT:     0:0-9 |   int d2
// CHECK-NEXT:         0 |   long d3
// CHECK-NEXT:           | [sizeof=8, align=8]

union u1 {
  unsigned int        :0;
  short               a;
} U1;
// CHECK:              0 | union u1
// CHECK-NEXT:       0:- |   unsigned int
// CHECK-NEXT:         0 |   short a
// CHECK-NEXT:           | [sizeof=4, align=4]

union u2 {
  long      :0;
  short      a;
} U2;
// CHECK:              0 | union u2
// CHECK-NEXT:       0:- |   long
// CHECK-NEXT:         0 |   short a
// CHECK-NEXT:           | [sizeof=8, align=8]

union u3 {
  unsigned char :0;
  unsigned short :0;
} U3;
// CHECK:              0 | union u3
// CHECK-NEXT:       0:- |   unsigned char
// CHECK-NEXT:       0:- |   unsigned short
// CHECK-NEXT:           | [sizeof=0, align=4]
