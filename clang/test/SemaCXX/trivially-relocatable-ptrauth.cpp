// RUN: %clang_cc1 -triple arm64 -fptrauth-calls -fptrauth-intrinsics -std=c++26 -verify %s

// This test intentionally does not enable the global address discrimination
// of vtable pointers. This lets us configure them with different schemas
// and verify that we're correctly tracking the existence of address discrimination

// expected-no-diagnostics

struct NonAddressDiscPtrauth {
  void * __ptrauth(1, 0, 1234) p;
};

static_assert(__builtin_is_cpp_trivially_relocatable(NonAddressDiscPtrauth));

struct AddressDiscPtrauth {
  void * __ptrauth(1, 1, 1234) p;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscPtrauth));

struct MultipleBaseClasses : NonAddressDiscPtrauth, AddressDiscPtrauth {

};

static_assert(!__builtin_is_cpp_trivially_relocatable(MultipleBaseClasses));

struct MultipleMembers1 {
   NonAddressDiscPtrauth field0;
   AddressDiscPtrauth field1;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(MultipleMembers1));

struct MultipleMembers2 {
   NonAddressDiscPtrauth field0;
   NonAddressDiscPtrauth field1;
};

static_assert(__builtin_is_cpp_trivially_relocatable(MultipleMembers2));

struct UnionOfPtrauth {
    union {
        NonAddressDiscPtrauth field0;
        AddressDiscPtrauth field1;
    } u;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(UnionOfPtrauth));

struct UnionOfNonAddressDiscriminatedPtrauth {
  union {
    NonAddressDiscPtrauth p;
    int i;
  } u;
};

static_assert(__builtin_is_cpp_trivially_relocatable(UnionOfNonAddressDiscriminatedPtrauth));

struct UnionOfAddressDisriminatedPtrauth {
  union {
    AddressDiscPtrauth p;
    int i;
  } u;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(UnionOfAddressDisriminatedPtrauth));
