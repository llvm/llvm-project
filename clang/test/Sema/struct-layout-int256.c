// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-gnu %s

// Test struct layout, alignment, and padding with __int256.

// Basic alignment and size
_Static_assert(sizeof(__int256) == 32, "");
_Static_assert(_Alignof(__int256) == 16, "");
_Static_assert(sizeof(unsigned __int256) == 32, "");

// Struct with __int256 member
struct Basic {
  __int256 x;
};
_Static_assert(sizeof(struct Basic) == 32, "");
_Static_assert(_Alignof(struct Basic) == 16, "");

// Struct with padding before __int256
struct Padded {
  char a;
  __int256 x;
};
// 'a' at offset 0 (1 byte), 15 bytes padding, 'x' at offset 16
_Static_assert(sizeof(struct Padded) == 48, "");
_Static_assert(_Alignof(struct Padded) == 16, "");

// Struct with multiple __int256 members
struct Multi {
  __int256 x;
  __int256 y;
};
_Static_assert(sizeof(struct Multi) == 64, "");

// Nested struct
struct Nested {
  struct Basic inner;
  int z;
};
_Static_assert(sizeof(struct Nested) == 48, ""); // 32 + 4 + 12 padding

// Union with __int256
union U {
  __int256 x;
  char bytes[32];
  long long parts[4];
};
_Static_assert(sizeof(union U) == 32, "");
_Static_assert(_Alignof(union U) == 16, "");

// Array of __int256
struct ArrayMember {
  __int256 arr[2];
};
_Static_assert(sizeof(struct ArrayMember) == 64, "");

// Packed struct
struct __attribute__((packed)) Packed {
  char a;
  __int256 x;
};
_Static_assert(sizeof(struct Packed) == 33, "");
_Static_assert(_Alignof(struct Packed) == 1, "");

// Aligned struct override
struct __attribute__((aligned(64))) OverAligned {
  __int256 x;
};
_Static_assert(sizeof(struct OverAligned) == 64, "");
_Static_assert(_Alignof(struct OverAligned) == 64, "");

// expected-no-diagnostics
