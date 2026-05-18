// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -Wweakened-alignment -verify=itanium -DITANIUM %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -Wweakened-alignment -verify=msvc -DMSVC %s

// msvc-no-diagnostics

#pragma pack(2)
struct D1 {
  char m1;
  alignas(8) int m4; // itanium-warning {{explicit alignment of m4 (8 bytes) was weakened to 2 bytes}}
};

#ifdef ITANIUM
static_assert(alignof(D1) == 2, "");
#else
static_assert(alignof(D1) == 8, "");
#endif
#pragma pack()

struct D2 {
  char m1;
  alignas(16) int m4; 
} __attribute__((packed));
static_assert(alignof(D2) == 16, "");

#pragma pack(8)
struct D3 {
  char m1;
  alignas(4) int m4; 
};
static_assert(alignof(D3) == 4, "");
#pragma pack()

struct alignas(16) Base1 { // itanium-warning {{explicit alignment of Base1 (16 bytes) was weakened to 2 bytes}}
  int x;
};

#pragma pack(2)
struct Derived1 : Base1 {
  char y;
};
#pragma pack()
Derived1 d1;

#ifdef ITANIUM
static_assert(alignof(Derived1) == 2, "");
#else
static_assert(alignof(Derived1) == 16, "");
#endif

struct alignas(8) Base2 { // itanium-warning {{explicit alignment of Base2 (8 bytes) was weakened to 4 bytes}}
  int x;
};

#pragma pack(4)
struct Derived2 : Base2 {
  char y;
};
#pragma pack()
Derived2 d2;

#ifdef ITANIUM
static_assert(alignof(Derived2) == 4, "");
#else
static_assert(alignof(Derived2) == 8, "");
#endif

struct alignas(16) VBase1 { // itanium-warning {{explicit alignment of VBase1 (16 bytes) was weakened to 4 bytes}}
  int x;
};

#pragma pack(4)
struct VDerived1 : virtual VBase1 {
  char y;
};
#pragma pack()
VDerived1 vd1;

#ifdef ITANIUM
static_assert(alignof(VDerived1) == 4, "");
#else
static_assert(alignof(VDerived1) == 16, "");
#endif

struct alignas(4) VBase2 { 
  int x;
};

#pragma pack(8)
struct VDerived2 : virtual VBase2 {
  char y;
};
#pragma pack()
VDerived2 vd2;
static_assert(alignof(VDerived2) >= 4, "");
