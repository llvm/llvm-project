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
