// RUN: %clang_cc1 -triple arm64               -std=c++26 -Wno-deprecated-builtins \
// RUN:                                        -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fptrauth-calls -fptrauth-intrinsics \
// RUN:                                       -fptrauth-vtable-pointer-address-discrimination \
// RUN:                                       -std=c++26 -Wno-deprecated-builtins \
// RUN:                                       -fsyntax-only -verify %s

// expected-no-diagnostics

#ifdef __PTRAUTH__
#define PTRAUTH_ENABLED 1
#define NonAddressDiscriminatedVTablePtrAttr \
  [[clang::ptrauth_vtable_pointer(process_independent, no_address_discrimination, no_extra_discrimination)]]
#define AddressDiscriminatedVTablePtrAttr \
  [[clang::ptrauth_vtable_pointer(process_independent, address_discrimination, no_extra_discrimination)]]
#define ADDR_DISC_ENABLED true
#else
#define PTRAUTH_ENABLED 0
#define NonAddressDiscriminatedVTablePtrAttr
#define AddressDiscriminatedVTablePtrAttr
#define ADDR_DISC_ENABLED false
#define __ptrauth(...)
#endif


typedef int* __ptrauth(1,1,1) AddressDiscriminatedPtr;
typedef __UINT64_TYPE__ __ptrauth(1,1,1) AddressDiscriminatedInt64;
struct AddressDiscriminatedFields {
  AddressDiscriminatedPtr ptr;
};
struct AddressDiscriminatedFieldInBaseClass : AddressDiscriminatedFields {
  void *newfield;
};

struct NonAddressDiscriminatedVTablePtrAttr NonAddressDiscriminatedVTablePtr {
  virtual ~NonAddressDiscriminatedVTablePtr();
  void *i;
};

struct NonAddressDiscriminatedVTablePtrAttr NonAddressDiscriminatedVTablePtr2 {
  virtual ~NonAddressDiscriminatedVTablePtr2();
  void *j;
};

struct AddressDiscriminatedVTablePtrAttr AddressDiscriminatedVTablePtr {
  virtual ~AddressDiscriminatedVTablePtr();
  void *k;
};
struct NoAddressDiscriminatedBaseClasses : NonAddressDiscriminatedVTablePtr,
                                           NonAddressDiscriminatedVTablePtr2 {
  void *l;
};

struct AddressDiscriminatedPrimaryBase : AddressDiscriminatedVTablePtr,
                                         NonAddressDiscriminatedVTablePtr {
  void *l;
};
struct AddressDiscriminatedSecondaryBase : NonAddressDiscriminatedVTablePtr,
                                           AddressDiscriminatedVTablePtr {
  void *l;
};

struct EmbdeddedAddressDiscriminatedPolymorphicClass {
  AddressDiscriminatedVTablePtr field;
};

static_assert( __is_pod(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_pod(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_pod(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert(!__is_pod(AddressDiscriminatedFieldInBaseClass));
static_assert(!__is_pod(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_pod(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_pod(AddressDiscriminatedVTablePtr));
static_assert(!__is_pod(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_pod(AddressDiscriminatedPrimaryBase));
static_assert(!__is_pod(AddressDiscriminatedSecondaryBase));
static_assert(!__is_pod(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_standard_layout(AddressDiscriminatedPtr));
static_assert( __is_standard_layout(AddressDiscriminatedInt64));
static_assert( __is_standard_layout(AddressDiscriminatedFields));
static_assert(!__is_standard_layout(AddressDiscriminatedFieldInBaseClass));
static_assert(!__is_standard_layout(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_standard_layout(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_standard_layout(AddressDiscriminatedVTablePtr));
static_assert(!__is_standard_layout(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_standard_layout(AddressDiscriminatedPrimaryBase));
static_assert(!__is_standard_layout(AddressDiscriminatedSecondaryBase));
static_assert(!__is_standard_layout(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_move_constructor(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_move_constructor(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_constructor(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_move_constructor(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_constructor(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_move_constructor(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_move_constructor(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_move_constructor(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_copy(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_copy(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_copy(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_copy(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_copy(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_copy(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_copy(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_copy(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_assign(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_assign(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_assign(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_assign(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_assign(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_assign(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_assign(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_assign(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_move_assign(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_move_assign(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_assign(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_move_assign(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_assign(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_move_assign(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_move_assign(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_move_assign(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivial(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivial(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivial(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivial(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivial(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivial(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivial(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivial(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivially_copyable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivially_copyable(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_copyable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_copyable(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_copyable(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_copyable(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_copyable(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_copyable(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivially_equality_comparable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_equality_comparable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedFields));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedFieldInBaseClass));
static_assert(!__is_trivially_equality_comparable(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_equality_comparable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_equality_comparable(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_equality_comparable(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivially_relocatable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivially_relocatable(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_relocatable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_relocatable(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_relocatable(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_relocatable(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_relocatable(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_relocatable(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__builtin_is_cpp_trivially_relocatable(NonAddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_cpp_trivially_relocatable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_cpp_trivially_relocatable(NoAddressDiscriminatedBaseClasses));
static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscriminatedPrimaryBase));
static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscriminatedSecondaryBase));
static_assert(!__builtin_is_cpp_trivially_relocatable(EmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_bitwise_cloneable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(NonAddressDiscriminatedVTablePtr));
static_assert( __is_bitwise_cloneable(NonAddressDiscriminatedVTablePtr2));
static_assert( __is_bitwise_cloneable(AddressDiscriminatedVTablePtr) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(NoAddressDiscriminatedBaseClasses));
static_assert( __is_bitwise_cloneable(AddressDiscriminatedPrimaryBase) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedSecondaryBase) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(EmbdeddedAddressDiscriminatedPolymorphicClass) == !ADDR_DISC_ENABLED);

static_assert( __has_unique_object_representations(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_unique_object_representations(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_unique_object_representations(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_unique_object_representations(AddressDiscriminatedVTablePtr));
static_assert(!__has_unique_object_representations(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_unique_object_representations(AddressDiscriminatedPrimaryBase));
static_assert(!__has_unique_object_representations(AddressDiscriminatedSecondaryBase));
static_assert(!__has_unique_object_representations(EmbdeddedAddressDiscriminatedPolymorphicClass));

#define ASSIGNABLE_WRAPPER(Type) __is_trivially_assignable(Type&, Type)
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!ASSIGNABLE_WRAPPER(NonAddressDiscriminatedVTablePtr));
static_assert(!ASSIGNABLE_WRAPPER(NonAddressDiscriminatedVTablePtr2));
static_assert(!ASSIGNABLE_WRAPPER(AddressDiscriminatedVTablePtr));
static_assert(!ASSIGNABLE_WRAPPER(NoAddressDiscriminatedBaseClasses));
static_assert(!ASSIGNABLE_WRAPPER(AddressDiscriminatedPrimaryBase));
static_assert(!ASSIGNABLE_WRAPPER(AddressDiscriminatedSecondaryBase));
static_assert(!ASSIGNABLE_WRAPPER(EmbdeddedAddressDiscriminatedPolymorphicClass));

namespace GH159505 {
  class A {
    virtual void f();
  };

  template <int N> struct B {
    class C : A {
      A a[N];
    } d;
  };

  template <int N> struct C {
    void *__ptrauth(1,1,1) ptr[N];
    static_assert(PTRAUTH_ENABLED != __is_trivially_copyable(decltype(ptr)));
  };
  template <class T, bool isPtrauth> struct D {
    T ptr;
    static_assert(isPtrauth != __is_trivially_copyable(decltype(ptr)));
  };


  template <class T> using Ptr = T * __ptrauth(1,1,1);
  template <class T> void test() {
    static_assert(PTRAUTH_ENABLED != __is_trivially_copyable(Ptr<T>));
  }

  auto f = test<int>;
  static_assert(!__is_trivially_copyable(B<1>));
  static_assert(PTRAUTH_ENABLED != __is_trivially_copyable(C<1>));


  D<void *, false> d_void;
  D<void * __ptrauth(1,1,1), PTRAUTH_ENABLED> d_void_ptrauth;
}
