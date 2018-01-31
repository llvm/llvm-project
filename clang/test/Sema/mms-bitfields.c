// RUN: %clang_cc1 -mms-bitfields -fsyntax-only -verify -triple x86_64-apple-darwin9 %s
// expected-no-diagnostics

// The -mms-bitfields commandline parameter should behave the same
// as the ms_struct attribute.
struct
{
   int a : 1;
   short b : 1;
} t;

// MS pads out bitfields between different types.
static int arr[(sizeof(t) == 8) ? 1 : -1];

#pragma pack (push,1)

typedef unsigned int UINT32;

struct Inner {
  UINT32    A    :  1;
  UINT32    B    :  1;
  UINT32    C    :  1;
  UINT32    D    : 30;
} Inner;

#pragma pack (pop)

static int arr2[(sizeof(Inner) == 8) ? 1 : -1];
