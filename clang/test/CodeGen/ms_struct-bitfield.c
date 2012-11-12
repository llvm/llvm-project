// RUN: %clang_cc1 -emit-llvm-only  -triple x86_64-apple-darwin9 %s
// rdar://8823265

#define ATTR __attribute__((__ms_struct__))

struct
{
   char foo;
   long : 0;
   char bar;
} ATTR t1;

struct
{
   char foo;
   long : 0;
   char : 0;
   int : 0;
   char bar;
} ATTR t2;

struct
{
   char foo;
   long : 0;
   char : 0;
   int : 0;
   char bar;
   long : 0;
   char : 0;
} ATTR t3;

struct
{
   long : 0;
   char bar;
} ATTR t4;

struct
{
   long : 0;
   long : 0;
   char : 0;
   char bar;
} ATTR t5;

struct
{
   long : 0;
   long : 0;
   char : 0;
   char bar;
} ATTR t6;

struct
{
   char foo;
   long : 0;
   int : 0;
   char bar;
   char bar1;
   long : 0;
   char bar2;
   char bar3;
   char : 0;
   char bar4;
   char bar5;
   char : 0;
   char bar6;
   char bar7;
} ATTR t7;

struct
{
   long : 0;
   long : 0;
   char : 0;
} ATTR t8;

struct
{
   char foo;
   long : 0;
   int : 0;
   char bar;
   char bar1;
   long : 0;
   char bar2;
   char bar3;
   char : 0;
   char bar4;
   char bar5;
   char : 0;
   char bar6;
   char bar7;
   int  i1;
   char : 0;
   long : 0;
   char :4;
   char bar8;
   char : 0;
   char bar9;
   char bar10;
   int  i2;
   char : 0;
   long : 0;
   char :4;
} ATTR t9;

struct
{
   char foo: 8;
   long : 0;
   char bar;
} ATTR t10;

static int arr1[(sizeof(t1) == 2) -1];
static int arr2[(sizeof(t2) == 2) -1];
static int arr3[(sizeof(t3) == 2) -1];
static int arr4[(sizeof(t4) == 1) -1];
static int arr5[(sizeof(t5) == 1) -1];
static int arr6[(sizeof(t6) == 1) -1];
static int arr7[(sizeof(t7) == 9) -1];
static int arr8[(sizeof(t8) == 0) -1];
static int arr9[(sizeof(t9) == 28) -1];
static int arr10[(sizeof(t10) == 16) -1];

int main() {
  return 0;
}

