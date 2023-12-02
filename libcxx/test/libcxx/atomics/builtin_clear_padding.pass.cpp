//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated-volatile -Wno-dynamic-class-memaccess

#include <cassert>
#include <cstdio>
#include <cstring>
#include <new>

template <class T>
void print_bytes(const T* object) {
  auto size                        = sizeof(T);
  const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(object);
  size_t i;

  fprintf(stderr, "[ ");
  for (i = 0; i < size; i++) {
    fprintf(stderr, "%02x ", bytes[i]);
  }
  fprintf(stderr, "]\n");
}

template <class T>
void __builtin_clear_padding2(T t) {
  __builtin_clear_padding(t);
}

template <size_t A1, size_t A2, class T>
struct alignas(A1) BasicWithPadding {
  T x;
  alignas(A2) T y;
};

template <size_t A1, size_t A2, size_t N, class T>
struct alignas(A1) SpacedArrayMembers {
  T x[N];
  alignas(A2) char c;
  T y[N];
};

template <size_t A1, size_t A2, class T>
struct alignas(A1) PaddedPointerMembers {
  T* x;
  alignas(A2) T* y;
};

template <size_t A1, size_t A2, size_t A3, class T>
struct alignas(A1) ThreeMembers {
  T x;
  alignas(A2) T y;
  alignas(A3) T z;
};

template <class T>
struct Normal {
  T a;
  T b;
};

template <class T>
struct X {
  T x;
};

template <class T>
struct Z {
  T z;
};

template <size_t A, class T>
struct YZ : public Z<T> {
  alignas(A) T y;
};

template <size_t A1, size_t A2, class T>
struct alignas(A1) HasBase : public X<T>, public YZ<A2, T> {
  T a;
  alignas(A2) T b;
};

template <size_t A1, size_t A2, class T>
void testAllStructsForType(T a, T b, T c, T d) {
  // basic padding
  {
    using B = BasicWithPadding<A1, A2, T>;
    B basic1;
    memset(&basic1, 0, sizeof(B));
    basic1.x = a;
    basic1.y = b;
    B basic2;
    memset(&basic2, 42, sizeof(B));
    basic2.x = a;
    basic2.y = b;
    assert(memcmp(&basic1, &basic2, sizeof(B)) != 0);
    __builtin_clear_padding2(&basic2);
    assert(memcmp(&basic1, &basic2, sizeof(B)) == 0);
  }

  // spaced array
  {
    using A = SpacedArrayMembers<A1, A2, 2, T>;
    A arr1;
    memset(&arr1, 0, sizeof(A));
    arr1.x[0] = a;
    arr1.x[1] = b;
    arr1.y[0] = c;
    arr1.y[1] = d;
    A arr2;
    memset(&arr2, 42, sizeof(A));
    arr2.x[0] = a;
    arr2.x[1] = b;
    arr2.y[0] = c;
    arr2.y[1] = d;
    arr2.c    = 0;
    assert(memcmp(&arr1, &arr2, sizeof(A)) != 0);
    __builtin_clear_padding2(&arr2);
    assert(memcmp(&arr1, &arr2, sizeof(A)) == 0);
  }

  // pointer members
  {
    using P = PaddedPointerMembers<A1, A2, T>;
    P ptr1;
    memset(&ptr1, 0, sizeof(P));
    ptr1.x = &a;
    ptr1.y = &b;
    P ptr2;
    memset(&ptr2, 42, sizeof(P));
    ptr2.x = &a;
    ptr2.y = &b;
    assert(memcmp(&ptr1, &ptr2, sizeof(P)) != 0);
    __builtin_clear_padding2(&ptr2);
    assert(memcmp(&ptr1, &ptr2, sizeof(P)) == 0);
  }

  // three members
  {
    using Three = ThreeMembers<A1, A2, A2, T>;
    Three three1;
    memset(&three1, 0, sizeof(Three));
    three1.x = a;
    three1.y = b;
    three1.z = c;
    Three three2;
    memset(&three2, 42, sizeof(Three));
    three2.x = a;
    three2.y = b;
    three2.z = c;
    __builtin_clear_padding2(&three2);
    assert(memcmp(&three1, &three2, sizeof(Three)) == 0);
  }

  // Normal struct no padding
  {
    using N = Normal<T>;
    N normal1;
    memset(&normal1, 0, sizeof(N));
    normal1.a = a;
    normal1.b = b;
    N normal2;
    memset(&normal2, 42, sizeof(N));
    normal2.a = a;
    normal2.b = b;
    __builtin_clear_padding2(&normal2);
    assert(memcmp(&normal1, &normal2, sizeof(N)) == 0);
  }

  // base class
  {
    using H = HasBase<A1, A2, T>;
    H base1;
    memset(&base1, 0, sizeof(H));
    base1.a = a;
    base1.b = b;
    base1.x = c;
    base1.y = d;
    base1.z = a;
    H base2;
    memset(&base2, 42, sizeof(H));
    base2.a = a;
    base2.b = b;
    base2.x = c;
    base2.y = d;
    base2.z = a;
    assert(memcmp(&base1, &base2, sizeof(H)) != 0);
    __builtin_clear_padding2(&base2);
    assert(memcmp(&base1, &base2, sizeof(H)) == 0);
  }
}

struct UnsizedTail {
  int size;
  alignas(8) char buf[];

  UnsizedTail(int size) : size(size) {}
};

void otherStructTests() {
  // Unsized Tail
  {
    const size_t size1 = sizeof(UnsizedTail) + 4;
    char buff1[size1];
    char buff2[size1];
    memset(buff1, 0, size1);
    memset(buff2, 42, size1);
    auto* u1   = new (buff1) UnsizedTail(4);
    u1->buf[0] = 1;
    u1->buf[1] = 2;
    u1->buf[2] = 3;
    u1->buf[3] = 4;
    auto* u2   = new (buff2) UnsizedTail(4);
    u2->buf[0] = 1;
    u2->buf[1] = 2;
    u2->buf[2] = 3;
    u2->buf[3] = 4;
    assert(memcmp(u1, u2, sizeof(UnsizedTail)) != 0);
    __builtin_clear_padding2(u2);

    assert(memcmp(u1, u2, sizeof(UnsizedTail)) == 0);
  }

  // basic padding on the heap
  {
    using B      = BasicWithPadding<8, 4, char>;
    auto* basic1 = new B;
    memset(basic1, 0, sizeof(B));
    basic1->x    = 1;
    basic1->y    = 2;
    auto* basic2 = new B;
    memset(basic2, 42, sizeof(B));
    basic2->x = 1;
    basic2->y = 2;
    assert(memcmp(basic1, basic2, sizeof(B)) != 0);
    __builtin_clear_padding2(basic2);
    assert(memcmp(basic1, basic2, sizeof(B)) == 0);
    delete basic2;
    delete basic1;
  }

  // basic padding volatile on the heap
  {
    using B   = BasicWithPadding<8, 4, char>;
    B* basic3 = new B;
    memset(basic3, 0, sizeof(B));
    basic3->x = 1;
    basic3->y = 2;
    B* basic4 = new B;
    memset(basic4, 42, sizeof(B));
    basic4->x = 1;
    basic4->y = 2;
    assert(memcmp(basic3, basic4, sizeof(B)) != 0);
    __builtin_clear_padding2(const_cast<volatile B*>(basic4));
    __builtin_clear_padding2(basic4);
    assert(memcmp(basic3, basic4, sizeof(B)) == 0);
    delete basic4;
    delete basic3;
  }
}

struct Foo {
  int x;
  int y;
};

typedef float Float4Vec __attribute__((ext_vector_type(4)));
typedef float Float3Vec __attribute__((ext_vector_type(3)));

void primitiveTests() {
  // no padding
  {
    int i1 = 42, i2 = 42;
    __builtin_clear_padding2(&i1); // does nothing
    assert(i1 == 42);
    assert(memcmp(&i1, &i2, sizeof(int)) == 0);
  }

  // long double
  {
    long double d1, d2;
    memset(&d1, 42, sizeof(long double));
    memset(&d2, 0, sizeof(long double));

    d1 = 3.0L;
    d2 = 3.0L;

    __builtin_clear_padding2(&d1);
    assert(d1 == 3.0L);
    assert(memcmp(&d1, &d2, sizeof(long double)) == 0);
  }
}

void structTests() {
  // no_unique_address
  {
    struct S1 {
      int x;
      char c;
    };

    struct S2 {
      [[no_unique_address]] S1 s;
      bool b;
    };

    S2 s1, s2;
    memset(&s1, 42, sizeof(S2));
    memset(&s2, 0, sizeof(S2));

    s1.s.x = 4;
    s1.s.c = 'a';
    s1.b   = true;
    s2.s.x = 4;
    s2.s.c = 'a';
    s2.b   = true;

    assert(memcmp(&s1, &s2, sizeof(S2)) != 0);
    __builtin_clear_padding2(&s1);
    assert(s1.s.x == 4);
    assert(s1.s.c == 'a');
    assert(s1.b == true);

    assert(memcmp(&s1, &s2, sizeof(S2)) == 0);
  }

  // struct with long double
  {
    struct S {
      long double l;
      bool b;
    };

    S s1, s2;
    memset(&s1, 42, sizeof(S));
    memset(&s2, 0, sizeof(S));

    s1.l = 3.0L;
    s1.b = true;
    s2.l = 3.0L;
    s2.b = true;

    assert(memcmp(&s1, &s2, sizeof(S)) != 0);
    __builtin_clear_padding2(&s1);
    assert(s1.l == 3.0L);
    assert(s1.b == true);
    assert(memcmp(&s1, &s2, sizeof(S)) == 0);
  }

  // EBO
  {
    struct Empty {};
    struct B {
      int i;
    };
    struct S : Empty, B {
      bool b;
    };

    S s1, s2;
    memset(&s1, 42, sizeof(S));
    memset(&s2, 0, sizeof(S));

    s1.i = 4;
    s1.b = true;
    s2.i = 4;
    s2.b = true;

    assert(memcmp(&s1, &s2, sizeof(S)) != 0);
    __builtin_clear_padding2(&s1);
    assert(s1.i == 4);
    assert(s1.b == true);
    assert(memcmp(&s1, &s2, sizeof(S)) == 0);
  }

  // padding between bases
  {
    struct B1 {
      char c1;
    };
    struct B2 {
      alignas(4) char c2;
    };

    struct S : B1, B2 {};

    S s1, s2;
    memset(&s1, 42, sizeof(S));
    memset(&s2, 0, sizeof(S));

    s1.c1 = 'a';
    s1.c2 = 'b';
    s2.c1 = 'a';
    s2.c2 = 'b';

    assert(memcmp(&s1, &s2, sizeof(S)) != 0);
    __builtin_clear_padding2(&s1);
    assert(s1.c1 == 'a');
    assert(s1.c2 == 'b');
    assert(memcmp(&s1, &s2, sizeof(S)) == 0);
  }

  // padding after last base
  {
    struct B1 {
      char c1;
    };
    struct B2 {
      char c2;
    };

    struct S : B1, B2 {
      alignas(4) char c3;
    };

    S s1, s2;
    memset(&s1, 42, sizeof(S));
    memset(&s2, 0, sizeof(S));

    s1.c1 = 'a';
    s1.c2 = 'b';
    s1.c3 = 'c';
    s2.c1 = 'a';
    s2.c2 = 'b';
    s2.c3 = 'c';

    assert(memcmp(&s1, &s2, sizeof(S)) != 0);
    __builtin_clear_padding2(&s1);
    assert(s1.c1 == 'a');
    assert(s1.c2 == 'b');
    assert(s1.c3 == 'c');
    assert(memcmp(&s1, &s2, sizeof(S)) == 0);
  }

  // vtable
  {
    struct VirtualBase {
      unsigned int x;
      virtual int call() { return x; };
      virtual ~VirtualBase() = default;
    };

    struct NonVirtualBase {
      char y;
    };

    struct S : VirtualBase, NonVirtualBase {
      virtual int call() override { return 5; }
      bool z;
    };

    char buff1[sizeof(S)];
    char buff2[sizeof(S)];
    memset(buff1, 0, sizeof(S));
    memset(buff2, 42, sizeof(S));

    S* s1 = new (&buff1) S;
    S* s2 = new (&buff2) S;

    s1->x = 0xFFFFFFFF;
    s2->x = 0xFFFFFFFF;
    s1->y = 'a';
    s2->y = 'a';
    s1->z = true;
    s2->z = true;
    __builtin_clear_padding2(s2);
    assert(s2->x == 0xFFFFFFFF);
    assert(s2->y == 'a');
    assert(s2->z == true);
    assert(s2->call() == 5);
    assert(memcmp(s1, s2, sizeof(S)) == 0);
  }

  // multiple bases with vtable
  {
    struct VirtualBase1 {
      unsigned int x1;
      virtual int call1() { return x1; };
      virtual ~VirtualBase1() = default;
    };

    struct VirtualBase2 {
      unsigned int x2;
      virtual int call2() { return x2; };
      virtual ~VirtualBase2() = default;
    };

    struct VirtualBase3 {
      unsigned int x3;
      virtual int call3() { return x3; };
      virtual ~VirtualBase3() = default;
    };

    struct NonVirtualBase {
      char y;
    };

    struct S : VirtualBase1, VirtualBase2, NonVirtualBase, VirtualBase3 {
      virtual int call1() override { return 5; }
      bool z;
    };

    char buff1[sizeof(S)];
    char buff2[sizeof(S)];
    memset(buff1, 0, sizeof(S));
    memset(buff2, 42, sizeof(S));

    S* s1 = new (&buff1) S;
    S* s2 = new (&buff2) S;

    s1->x1 = 0xFFFFFFFF;
    s2->x1 = 0xFFFFFFFF;
    s1->x2 = 0xFAFAFAFA;
    s2->x2 = 0xFAFAFAFA;
    s1->x3 = 0xAAAAAAAA;
    s2->x3 = 0xAAAAAAAA;
    s1->y  = 'a';
    s2->y  = 'a';
    s1->z  = true;
    s2->z  = true;
    __builtin_clear_padding2(s2);
    assert(s2->x1 == 0xFFFFFFFF);
    assert(s2->x2 == 0xFAFAFAFA);
    assert(s2->x3 == 0xAAAAAAAA);
    assert(s2->y == 'a');
    assert(s2->z == true);
    assert(s2->call1() == 5);
    assert(memcmp(s1, s2, sizeof(S)) == 0);
  }

  // chain of bases with virtual functions
  {
    struct VirtualBase1 {
      unsigned int x1;
      virtual int call1() { return x1; };
      virtual ~VirtualBase1() = default;
    };

    struct VirtualBase2 : VirtualBase1 {
      unsigned int x2;
      virtual int call2() { return x2; };
      virtual ~VirtualBase2() = default;
    };

    struct VirtualBase3 : VirtualBase2 {
      unsigned int x3;
      virtual int call3() { return x3; };
      virtual ~VirtualBase3() = default;
    };

    struct NonVirtualBase {
      char y;
    };

    struct S : NonVirtualBase, VirtualBase3 {
      //virtual int call() override { return 5; }
      bool z;
    };

    char buff1[sizeof(S)];
    char buff2[sizeof(S)];
    memset(buff1, 0, sizeof(S));
    memset(buff2, 42, sizeof(S));
    S* s1 = new (&buff1) S;
    S* s2 = new (&buff2) S;

    s1->x1 = 0xFFFFFFFF;
    s2->x1 = 0xFFFFFFFF;
    s1->x2 = 0xFAFAFAFA;
    s2->x2 = 0xFAFAFAFA;
    s1->x3 = 0xAAAAAAAA;
    s2->x3 = 0xAAAAAAAA;
    s1->y  = 'a';
    s2->y  = 'a';
    s1->z  = true;
    s2->z  = true;
    __builtin_clear_padding2(s2);
    assert(memcmp(s1, s2, sizeof(S)) == 0);
  }

  // virtual inheritance
  {
    struct Base {
      int x;
    };
    struct D1 : virtual Base {
      int d1;
      bool b1;
    };
    struct D2 : virtual Base {
      int d2;
      bool b2;
    };

    struct S : D1, D2 {
      bool s;
    };

    char buff1[sizeof(S)];
    char buff2[sizeof(S)];
    memset(buff1, 0, sizeof(S));
    memset(buff2, 42, sizeof(S));
    S* s1 = new (&buff1) S;
    S* s2 = new (&buff2) S;

    s1->x  = 0xFFFFFFFF;
    s2->x  = 0xFFFFFFFF;
    s1->d1 = 0xFAFAFAFA;
    s2->d1 = 0xFAFAFAFA;
    s1->d2 = 0xAAAAAAAA;
    s2->d2 = 0xAAAAAAAA;
    s1->b1 = true;
    s2->b1 = true;
    s1->b2 = true;
    s2->b2 = true;
    s1->s  = true;
    s2->s  = true;
    __builtin_clear_padding2(s2);
    assert(memcmp(s1, s2, sizeof(S)) == 0);
  }

  // bit fields
  {
    struct S {
      // will usually occupy 2 bytes:
      unsigned char b1 : 3; // 1st 3 bits (in 1st byte) are b1
      unsigned char    : 2; // next 2 bits (in 1st byte) are blocked out as unused
      unsigned char b2 : 6; // 6 bits for b2 - doesn't fit into the 1st byte => starts a 2nd
      unsigned char b3 : 2; // 2 bits for b3 - next (and final) bits in the 2nd byte
    };

    S s1, s2;
    memset(&s1, 0, sizeof(S));
    memset(&s2, 42, sizeof(S));

    s1.b1 = 5;
    s2.b1 = 5;
    s1.b2 = 27;
    s2.b2 = 27;
    s1.b3 = 3;
    s2.b3 = 3;
    __builtin_clear_padding(&s2);
    print_bytes(&s1);
    print_bytes(&s2);
    //TODO
    //assert(memcmp(&s1, &s2, sizeof(S)) == 0);
  }

  testAllStructsForType<32, 16, char>(11, 22, 33, 44);
  testAllStructsForType<64, 32, char>(4, 5, 6, 7);
  testAllStructsForType<32, 16, volatile char>(11, 22, 33, 44);
  testAllStructsForType<64, 32, volatile char>(4, 5, 6, 7);
  testAllStructsForType<32, 16, int>(0, 1, 2, 3);
  testAllStructsForType<64, 32, int>(4, 5, 6, 7);
  testAllStructsForType<32, 16, volatile int>(0, 1, 2, 3);
  testAllStructsForType<64, 32, volatile int>(4, 5, 6, 7);
  testAllStructsForType<32, 16, double>(0, 1, 2, 3);
  testAllStructsForType<64, 32, double>(4, 5, 6, 7);
  testAllStructsForType<32, 16, _BitInt(28)>(0, 1, 2, 3);
  testAllStructsForType<64, 32, _BitInt(28)>(4, 5, 6, 7);
  testAllStructsForType<32, 16, _BitInt(60)>(0, 1, 2, 3);
  testAllStructsForType<64, 32, _BitInt(60)>(4, 5, 6, 7);
  testAllStructsForType<32, 16, _BitInt(64)>(0, 1, 2, 3);
  testAllStructsForType<64, 32, _BitInt(64)>(4, 5, 6, 7);
  testAllStructsForType<32, 16, Foo>(Foo{1, 2}, Foo{3, 4}, Foo{1, 2}, Foo{3, 4});
  testAllStructsForType<64, 32, Foo>(Foo{1, 2}, Foo{3, 4}, Foo{1, 2}, Foo{3, 4});
  testAllStructsForType<256, 128, Float3Vec>(0, 1, 2, 3);
  testAllStructsForType<128, 128, Float3Vec>(4, 5, 6, 7);
  testAllStructsForType<256, 128, Float4Vec>(0, 1, 2, 3);
  testAllStructsForType<128, 128, Float4Vec>(4, 5, 6, 7);

  otherStructTests();
}

void unionTests() {
  // different length, do not clear object repr bits of non-active member
  {
    union u {
      int i;
      char c;
    };

    u u1, u2;
    memset(&u1, 42, sizeof(u));
    memset(&u2, 42, sizeof(u));
    u1.c = '4';
    u2.c = '4';

    __builtin_clear_padding2(&u1); // should have no effect
    assert(u1.c == '4');

    assert(memcmp(&u1, &u2, sizeof(u)) == 0);
  }

  // tail padding of longest member
  {
    struct s {
      alignas(8) char c1;
    };

    union u {
      s s1;
      char c2;
    };

    u u1, u2;
    memset(&u1, 42, sizeof(u));
    memset(&u2, 0, sizeof(u));

    u1.s1.c1 = '4';
    u2.s1.c1 = '4';

    assert(memcmp(&u1, &u2, sizeof(u)) != 0);
    __builtin_clear_padding2(&u1);
    assert(u1.s1.c1 == '4');
    assert(memcmp(&u1, &u2, sizeof(u)) == 0);
  }
}

void arrayTests() {
  // no padding
  {
    int i1[2] = {1, 2};
    int i2[2] = {1, 2};

    __builtin_clear_padding2(&i1);
    assert(i1[0] == 1);
    assert(i1[1] == 2);
    assert(memcmp(&i1, &i2, 2 * sizeof(int)) == 0);
  }

  // long double
  {
    long double d1[2], d2[2];
    memset(&d1, 42, 2 * sizeof(long double));
    memset(&d2, 0, 2 * sizeof(long double));

    d1[0] = 3.0L;
    d1[1] = 4.0L;
    d2[0] = 3.0L;
    d2[1] = 4.0L;

    __builtin_clear_padding2(&d1);
    assert(d1[0] == 3.0L);
    assert(d2[1] == 4.0L);
    assert(memcmp(&d1, &d2, 2 * sizeof(long double)) == 0);
  }

  // struct
  {
    struct S {
      int i1;
      char c1;
      int i2;
      char c2;
    };

    S s1[2], s2[2];
    memset(&s1, 42, 2 * sizeof(S));
    memset(&s2, 0, 2 * sizeof(S));

    s1[0].i1 = 1;
    s1[0].c1 = 'a';
    s1[0].i2 = 2;
    s1[0].c2 = 'b';
    s1[1].i1 = 3;
    s1[1].c1 = 'c';
    s1[1].i2 = 4;
    s1[1].c2 = 'd';

    s2[0].i1 = 1;
    s2[0].c1 = 'a';
    s2[0].i2 = 2;
    s2[0].c2 = 'b';
    s2[1].i1 = 3;
    s2[1].c1 = 'c';
    s2[1].i2 = 4;
    s2[1].c2 = 'd';

    assert(memcmp(&s1, &s2, 2 * sizeof(S)) != 0);
    __builtin_clear_padding2(&s1);

    assert(s1[0].i1 == 1);
    assert(s1[0].c1 == 'a');
    assert(s1[0].i2 == 2);
    assert(s1[0].c2 == 'b');
    assert(s1[1].i1 == 3);
    assert(s1[1].c1 == 'c');
    assert(s1[1].i2 == 4);
    assert(s1[1].c2 == 'd');
    assert(memcmp(&s1, &s2, 2 * sizeof(S)) == 0);
  }
}

int main(int, const char**) {
  primitiveTests();
  unionTests();
  structTests();
  arrayTests();

  return 0;
}
