// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s
// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple i386-pc-win32 -fms-compatibility %s

__int8 x1  = 3i8;
__int16 x2 = 4i16;
__int32 x3 = 5i32;
__int64 x5 = 0x42i64;
__int64 x6 = 0x42I64;

__int64 y = 0x42i64u;  // expected-error {{invalid suffix}}
__int64 w = 0x43ui64; 
__int64 z = 9Li64;  // expected-error {{invalid suffix}}
__int64 q = 10lli64;  // expected-error {{invalid suffix}}

__complex double c1 = 1i;
__complex double c2 = 1.0i;
__complex float c3 = 1.0if;

// radar 7562363
#define ULLONG_MAX 0xffffffffffffffffui64
#define UINT 0xffffffffui32
#define USHORT 0xffffui16
#define UCHAR 0xffui8

void a(void) {
	unsigned long long m = ULLONG_MAX;
	unsigned int n = UINT;
        unsigned short s = USHORT;
        unsigned char c = UCHAR;
}

void pr_7968(void)
{
  int var1 = 0x1111111e+1;
  int var2 = 0X1111111e+1;
  int var3 = 0xe+1;
  int var4 = 0XE+1;

  int var5=    0\
x1234e+1;

  int var6=
  /*expected-warning {{backslash and newline separated by space}} */    0\       
x1234e+1;                      
}

