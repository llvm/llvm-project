// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s


static const unsigned long long scull = 0;
void static_int()
{
    *(int*)scull = 0; // expected-warning{{Dereference of null pointer}}
}

const unsigned long long cull = 0;
void const_int()
{
    *(int*)cull = 0; // expected-warning{{Dereference of null pointer}}
}

static int * const spc = 0;
void static_ptr()
{
    *spc = 0; // expected-warning{{Dereference of null pointer}}
}

int * const pc = 0;
void const_ptr()
{
    *pc = 0; // expected-warning{{Dereference of null pointer}}
}

const unsigned long long cull_nonnull = 4;
void nonnull_int()
{
    *(int*)(cull_nonnull - 4) = 0; // expected-warning{{Dereference of null pointer}}
}

int * const pc_nonnull = (int*)sizeof(int);
void nonnull_ptr()
{
    *(pc_nonnull - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

int * const constcast = const_cast<int * const>((int*)sizeof(int));
void cast1()
{
    *(constcast - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

int * const recast = reinterpret_cast<int*>(sizeof(int));
void cast2()
{
    *(recast - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

int * const staticcast = static_cast<int * const>((int*)sizeof(int));
void cast3()
{
    *(staticcast - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

struct Foo { int a; };
Foo * const dyncast = dynamic_cast<Foo * const>((Foo*)sizeof(Foo));
void cast4()
{
    // Do not handle dynamic_cast for now, because it may change the pointer value.
    (dyncast - 1)->a = 0; // no-warning
}

typedef int * const intptrconst;
int * const funccast = intptrconst(sizeof(int));
void cast5()
{
    *(funccast - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

struct S1
{
    int * p;
};
const S1 s1 = {
    .p = (int*)sizeof(int)
};
void conststruct()
{
    *(s1.p - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

struct S2
{
    int * const p;
};
S2 s2 = {
    .p = (int*)sizeof(int)
};
void constfield()
{
    *(s2.p - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

int * const parr[1] = { (int*)sizeof(int) };
void constarr()
{
    *(parr[0] - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

struct S3
{
    int * p = (int*)sizeof(int);
};
void recordinit()
{
    S3 s3;
    *(s3.p - 1) = 0; // expected-warning{{Dereference of null pointer}}
}

extern int ext_int;

void update_original_declaration() {
  ext_int = 2;
}

extern int ext_int;

int test_redeclaration() {
  ext_int = 1;
  update_original_declaration();
  int int_int = 3 / (ext_int - 1); // no-warning
  return int_int / (ext_int - 2); // expected-warning{{Division by zero}}
}
