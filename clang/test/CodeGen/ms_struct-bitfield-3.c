// RUN: %clang_cc1 -emit-llvm-only  -triple i386-apple-darwin9 %s
// rdar://8823265

#define ATTR __attribute__((__ms_struct__))

struct _struct_0
{
  int  member_0   : 25 ;
  short  member_1   : 6 ;
  char  member_2   : 2 ;
  unsigned  short  member_3   : 1 ;
  unsigned  char  member_4   : 7 ;
  short  member_5   : 16 ;
  int  : 0 ;
  char  member_7  ;

} ATTR;

typedef struct _struct_0 struct_0;

#define size_struct_0 20

struct_0 test_struct_0 = { 18557917, 17, 3, 0, 80, 6487, 93 };
static int a[(size_struct_0 == sizeof (struct_0)) -1];

struct _struct_1 {
  int d;
  unsigned char a;
  unsigned short b:7;
  char c;
} ATTR;

typedef struct _struct_1 struct_1;

#define size_struct_1 12

struct_1 test_struct_1 = { 18557917, 'a', 3, 'b' };

static int a1[(size_struct_1 == sizeof (struct_1)) -1];

struct ten {
  long long a:3;
  long long b:3;
  char c;
} __attribute__ ((ms_struct));

#define size_struct_2 16

static int a2[(size_struct_2 == sizeof (struct ten)) -1];
