// RUN: %clang_cc1 %s -fsyntax-only -verify -fborland-extensions
// expected-no-diagnostics

// Borland extensions

// 1. test  -fborland-extensions
int dummy_function() { return 0; }

// 2. test __pascal
int _pascal f2();

float __pascal gi2(int, int); 
template<typename T> T g2(T (__pascal * const )(int, int)) { return 0; }

struct M {
    int __pascal addP();
    float __pascal subtractP(); 
};
template<typename T> int h2(T (__pascal M::* const )()) { return 0; }
void m2() {
    int i; float f;
    i = f2();
    f = gi2(2, i);
    f = g2(gi2);
    i = h2<int>(&M::addP);
    f = h2(&M::subtractP);
} 

// 3. test other calling conventions
int _cdecl fa3();
int _fastcall fc3();
int _stdcall fd3();

// 4. test __uuidof()
typedef struct _GUID {
     unsigned long  Data1;
     unsigned short Data2;
     unsigned short Data3;
     unsigned char  Data4[ 8 ];
} GUID;

struct __declspec(uuid("{12345678-1234-1234-1234-123456789abc}")) Foo;
struct Data {
     GUID const* Guid;
};

void t4() {
    unsigned long  data;

    const GUID guid_inl = __uuidof(Foo);
    Data ata1 = { &guid_inl};
    data = ata1.Guid->Data1;
}

