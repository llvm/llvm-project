// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify -triple x86_64-apple-darwin9 %s

#pragma ms_struct on

#pragma ms_struct off

#pragma ms_struct reset

#pragma ms_struct // expected-warning {{incorrect use of '#pragma ms_struct on|off' - ignored}}

#pragma ms_struct on top of spaghetti  // expected-warning {{extra tokens at end of '#pragma ms_struct' - ignored}}

struct foo
{
  int a;
  int b;
  char c;
};


struct {
                   unsigned long bf_1 : 12;
                   unsigned long : 0;
                   unsigned long bf_2 : 12;
} __attribute__((__ms_struct__)) t1;

struct S {
		   double __attribute__((ms_struct)) d;	// expected-warning {{'ms_struct' attribute only applies to structs, unions, and classes}}
                   unsigned long bf_1 : 12;
                   unsigned long : 0;
                   unsigned long bf_2 : 12;
} __attribute__((ms_struct)) t2;

enum
{
  A = 0,
  B,
  C
} __attribute__((ms_struct)) e1; // expected-warning {{'ms_struct' attribute only applies to}}

// rdar://10513599
#pragma ms_struct on

typedef struct
{
void *pv;
int l;
} Foo;

typedef struct
{
void *pv1;
Foo foo;
unsigned short fInited : 1;
void *pv2;
} PackOddity;

#pragma ms_struct off

static int arr[sizeof(PackOddity) == 40 ? 1 : -1];

struct __declspec(ms_struct) bad { // expected-warning {{__declspec attribute 'ms_struct' is not supported}}
};
