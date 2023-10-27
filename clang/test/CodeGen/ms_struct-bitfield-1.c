// RUN: %clang_cc1 -emit-llvm-only  -triple x86_64-apple-darwin9 %s

#define ATTR __attribute__((__ms_struct__))

struct {
  unsigned int bf_1 : 12;
  unsigned int : 0;
  unsigned int bf_2 : 12;
} ATTR t1;
static int a1[(sizeof(t1) == 8) -1];

struct
{
  char foo : 4;
  short : 0;
  char bar;
} ATTR t2;
static int a2[(sizeof(t2) == 4) -1];

#pragma ms_struct on
struct
{
  char foo : 4;
  short : 0;
  char bar;
} t3;
#pragma ms_struct off
static int a3[(sizeof(t3) == 4) -1];

struct
{
  char foo : 6;
  long : 0;
} ATTR t4;
static int a4[(sizeof(t4) == 8) -1];

struct
{
  char foo : 4;
  short : 0;
  char bar : 8;
} ATTR t5;
static int a5[(sizeof(t5) == 4) -1];

struct
{
  char foo : 4;
  short : 0;
  long  : 0;
  char bar;
} ATTR t6;
static int a6[(sizeof(t6) == 4) -1];

struct
{
  char foo : 4;
  long  : 0;
  short : 0;
  char bar;
} ATTR t7;
static int a7[(sizeof(t7) == 16) -1];

struct
{
  char foo : 4;
  short : 0;
  long  : 0;
  char bar:7;
} ATTR t8;
static int a8[(sizeof(t8) == 4) -1];

struct
{
  char foo : 4;
  long  : 0;
  short : 0;
  char bar: 8;
} ATTR t9;
static int a9[(sizeof(t9) == 16) -1];

struct
{
  char foo : 4;
  char : 0;
  short : 0;
  int : 0;
  long  :0;
  char bar;
} ATTR t10;
static int a10[(sizeof(t10) == 2) -1];
