// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-macosx10.8.0 -fms-extensions -verify

typedef struct _GUID
{
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
} GUID;

struct __declspec(uuid("87654321-4321-4321-4321-ba0987654321")) S { };

GUID g = __uuidof(S);  // expected-error {{__uuidof codegen is not supported on this architecture}}
