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

struct [[clang::ptrauth_vtable_pointer(process_independent,address_discrimination,no_extra_discrimination)]] Polymorphic trivially_relocatable_if_eligible {
  virtual ~Polymorphic();
};

struct Foo : Polymorphic {
  Foo(const Foo&);
  ~Foo();
};


static_assert(!__builtin_is_cpp_trivially_relocatable(Polymorphic));

struct [[clang::ptrauth_vtable_pointer(process_independent,no_address_discrimination,no_extra_discrimination)]] NonAddressDiscriminatedPolymorphic trivially_relocatable_if_eligible {
  virtual ~NonAddressDiscriminatedPolymorphic();
};

static_assert(__builtin_is_cpp_trivially_relocatable(NonAddressDiscriminatedPolymorphic));


struct PolymorphicMembers {
    Polymorphic field;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(PolymorphicMembers));

struct UnionOfPolymorphic {
  union trivially_relocatable_if_eligible {
    Polymorphic p;
    int i;
  } u;
};

static_assert(!__builtin_is_cpp_trivially_relocatable(UnionOfPolymorphic));


struct UnionOfNonAddressDiscriminatedPolymorphic {
  union trivially_relocatable_if_eligible {
    NonAddressDiscriminatedPolymorphic p;
    int i;
  } u;
};
static_assert(!__builtin_is_cpp_trivially_relocatable(UnionOfNonAddressDiscriminatedPolymorphic));

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
