// RUN: %clang_cc1 -triple arm64               -std=c++26 -Wno-deprecated-builtins \
// RUN:                                        -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fptrauth-calls -fptrauth-intrinsics \
// RUN:                                       -fptrauth-vtable-pointer-address-discrimination \
// RUN:                                       -std=c++26 -Wno-deprecated-builtins \
// RUN:                                       -fsyntax-only -verify %s

// expected-no-diagnostics

#ifdef __PTRAUTH__

#define NonAddressDiscriminatedVTablePtrAttr \
  [[clang::ptrauth_vtable_pointer(process_independent, no_address_discrimination, no_extra_discrimination)]]
#define AddressDiscriminatedVTablePtrAttr \
  [[clang::ptrauth_vtable_pointer(process_independent, address_discrimination, no_extra_discrimination)]]
#define ADDR_DISC_ENABLED true
#else
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
struct RelocatableAddressDiscriminatedFields trivially_relocatable_if_eligible {
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

struct NonAddressDiscriminatedVTablePtrAttr RelocatableNonAddressDiscriminatedVTablePtr trivially_relocatable_if_eligible {
  virtual ~RelocatableNonAddressDiscriminatedVTablePtr();
  void *i;
};

struct NonAddressDiscriminatedVTablePtrAttr RelocatableNonAddressDiscriminatedVTablePtr2 trivially_relocatable_if_eligible {
  virtual ~RelocatableNonAddressDiscriminatedVTablePtr2();
  void *j;
};

struct AddressDiscriminatedVTablePtrAttr AddressDiscriminatedVTablePtr {
  virtual ~AddressDiscriminatedVTablePtr();
  void *k;
};

struct AddressDiscriminatedVTablePtrAttr RelocatableAddressDiscriminatedVTablePtr trivially_relocatable_if_eligible {
  virtual ~RelocatableAddressDiscriminatedVTablePtr();
  void *k;
};

struct NoAddressDiscriminatedBaseClasses : NonAddressDiscriminatedVTablePtr,
                                           NonAddressDiscriminatedVTablePtr2 {
  void *l;
};

struct RelocatableNoAddressDiscriminatedBaseClasses trivially_relocatable_if_eligible :
                                           NonAddressDiscriminatedVTablePtr,
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

struct RelocatableAddressDiscriminatedPrimaryBase : RelocatableAddressDiscriminatedVTablePtr,
                                         RelocatableNonAddressDiscriminatedVTablePtr {
  void *l;
};
struct RelocatableAddressDiscriminatedSecondaryBase : RelocatableNonAddressDiscriminatedVTablePtr,
                                           RelocatableAddressDiscriminatedVTablePtr {
  void *l;
};
struct EmbdeddedAddressDiscriminatedPolymorphicClass {
  AddressDiscriminatedVTablePtr field;
};
struct RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass trivially_relocatable_if_eligible {
  AddressDiscriminatedVTablePtr field;
};

static_assert( __is_pod(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_pod(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_pod(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_pod(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert(!__is_pod(AddressDiscriminatedFieldInBaseClass));
static_assert(!__is_pod(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_pod(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_pod(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__is_pod(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__is_pod(AddressDiscriminatedVTablePtr));
static_assert(!__is_pod(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__is_pod(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_pod(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__is_pod(AddressDiscriminatedPrimaryBase));
static_assert(!__is_pod(AddressDiscriminatedSecondaryBase));
static_assert(!__is_pod(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__is_pod(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__is_pod(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__is_pod(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_standard_layout(AddressDiscriminatedPtr));
static_assert( __is_standard_layout(AddressDiscriminatedInt64));
static_assert( __is_standard_layout(AddressDiscriminatedFields));
static_assert( __is_standard_layout(RelocatableAddressDiscriminatedFields));
static_assert(!__is_standard_layout(AddressDiscriminatedFieldInBaseClass));
static_assert(!__is_standard_layout(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_standard_layout(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_standard_layout(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__is_standard_layout(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__is_standard_layout(AddressDiscriminatedVTablePtr));
static_assert(!__is_standard_layout(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__is_standard_layout(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_standard_layout(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__is_standard_layout(AddressDiscriminatedPrimaryBase));
static_assert(!__is_standard_layout(AddressDiscriminatedSecondaryBase));
static_assert(!__is_standard_layout(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__is_standard_layout(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__is_standard_layout(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__is_standard_layout(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_move_constructor(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_constructor(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_move_constructor(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_constructor(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_move_constructor(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_constructor(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_move_constructor(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_constructor(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_constructor(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_move_constructor(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_move_constructor(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_move_constructor(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_move_constructor(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_move_constructor(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_move_constructor(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__has_trivial_move_constructor(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_copy(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_copy(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_copy(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_copy(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_copy(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_copy(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_copy(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_copy(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_copy(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_copy(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_copy(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_copy(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_copy(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_copy(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_copy(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__has_trivial_copy(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_assign(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_assign(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_assign(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_assign(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_assign(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_assign(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_assign(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_assign(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_assign(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_assign(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_assign(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_assign(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_assign(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_assign(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_assign(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__has_trivial_assign(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __has_trivial_move_assign(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_trivial_move_assign(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_trivial_move_assign(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_assign(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_move_assign(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_assign(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__has_trivial_move_assign(AddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_assign(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__has_trivial_move_assign(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_move_assign(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__has_trivial_move_assign(AddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_move_assign(AddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_move_assign(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__has_trivial_move_assign(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__has_trivial_move_assign(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__has_trivial_move_assign(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivial(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivial(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivial(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivial(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivial(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivial(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivial(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivial(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__is_trivial(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivial(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivial(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivial(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivial(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__is_trivial(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__is_trivial(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__is_trivial(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivially_copyable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_copyable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivially_copyable(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_copyable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_copyable(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_copyable(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_copyable(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_copyable(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_copyable(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_copyable(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_copyable(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_copyable(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_copyable(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_copyable(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_copyable(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__is_trivially_copyable(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivially_equality_comparable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_equality_comparable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedFields));
static_assert(!__is_trivially_equality_comparable(RelocatableAddressDiscriminatedFields));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedFieldInBaseClass));
static_assert(!__is_trivially_equality_comparable(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_equality_comparable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_equality_comparable(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_equality_comparable(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_equality_comparable(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_equality_comparable(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_equality_comparable(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_equality_comparable(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_equality_comparable(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_equality_comparable(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_equality_comparable(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__is_trivially_equality_comparable(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_trivially_relocatable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_trivially_relocatable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__is_trivially_relocatable(NonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_relocatable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_relocatable(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_relocatable(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__is_trivially_relocatable(AddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_relocatable(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__is_trivially_relocatable(NoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_relocatable(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__is_trivially_relocatable(AddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_relocatable(AddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_relocatable(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__is_trivially_relocatable(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__is_trivially_relocatable(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__is_trivially_relocatable(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__builtin_is_cpp_trivially_relocatable(NonAddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_cpp_trivially_relocatable(NonAddressDiscriminatedVTablePtr2));
static_assert( __builtin_is_cpp_trivially_relocatable(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert( __builtin_is_cpp_trivially_relocatable(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscriminatedVTablePtr));
static_assert( __builtin_is_cpp_trivially_relocatable(RelocatableAddressDiscriminatedVTablePtr) == !ADDR_DISC_ENABLED);
static_assert(!__builtin_is_cpp_trivially_relocatable(NoAddressDiscriminatedBaseClasses));
static_assert(!__builtin_is_cpp_trivially_relocatable(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscriminatedPrimaryBase));
static_assert(!__builtin_is_cpp_trivially_relocatable(AddressDiscriminatedSecondaryBase));
static_assert( __builtin_is_cpp_trivially_relocatable(RelocatableAddressDiscriminatedPrimaryBase) == !ADDR_DISC_ENABLED);
static_assert( __builtin_is_cpp_trivially_relocatable(RelocatableAddressDiscriminatedSecondaryBase) == !ADDR_DISC_ENABLED);
static_assert(!__builtin_is_cpp_trivially_relocatable(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__builtin_is_cpp_trivially_relocatable(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __builtin_is_replaceable(AddressDiscriminatedPtr));
static_assert( __builtin_is_replaceable(AddressDiscriminatedInt64));
static_assert( __builtin_is_replaceable(AddressDiscriminatedFields));
static_assert( __builtin_is_replaceable(RelocatableAddressDiscriminatedFields));
static_assert( __builtin_is_replaceable(AddressDiscriminatedFieldInBaseClass));
static_assert(!__builtin_is_replaceable(NonAddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_replaceable(NonAddressDiscriminatedVTablePtr2));
static_assert(!__builtin_is_replaceable(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_replaceable(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__builtin_is_replaceable(AddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_replaceable(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__builtin_is_replaceable(NoAddressDiscriminatedBaseClasses));
static_assert(!__builtin_is_replaceable(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__builtin_is_replaceable(AddressDiscriminatedPrimaryBase));
static_assert(!__builtin_is_replaceable(AddressDiscriminatedSecondaryBase));
static_assert(!__builtin_is_replaceable(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__builtin_is_replaceable(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__builtin_is_replaceable(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__builtin_is_replaceable(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

static_assert( __is_bitwise_cloneable(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(NonAddressDiscriminatedVTablePtr));
static_assert( __is_bitwise_cloneable(NonAddressDiscriminatedVTablePtr2));
static_assert( __is_bitwise_cloneable(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert( __is_bitwise_cloneable(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert( __is_bitwise_cloneable(AddressDiscriminatedVTablePtr) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(RelocatableAddressDiscriminatedVTablePtr) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(NoAddressDiscriminatedBaseClasses));
static_assert( __is_bitwise_cloneable(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert( __is_bitwise_cloneable(AddressDiscriminatedPrimaryBase) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(AddressDiscriminatedSecondaryBase) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(RelocatableAddressDiscriminatedPrimaryBase) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(RelocatableAddressDiscriminatedSecondaryBase) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(EmbdeddedAddressDiscriminatedPolymorphicClass) == !ADDR_DISC_ENABLED);
static_assert( __is_bitwise_cloneable(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass) == !ADDR_DISC_ENABLED);

static_assert( __has_unique_object_representations(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( __has_unique_object_representations(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!__has_unique_object_representations(NonAddressDiscriminatedVTablePtr));
static_assert(!__has_unique_object_representations(NonAddressDiscriminatedVTablePtr2));
static_assert(!__has_unique_object_representations(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!__has_unique_object_representations(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!__has_unique_object_representations(AddressDiscriminatedVTablePtr));
static_assert(!__has_unique_object_representations(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!__has_unique_object_representations(NoAddressDiscriminatedBaseClasses));
static_assert(!__has_unique_object_representations(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!__has_unique_object_representations(AddressDiscriminatedPrimaryBase));
static_assert(!__has_unique_object_representations(AddressDiscriminatedSecondaryBase));
static_assert(!__has_unique_object_representations(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!__has_unique_object_representations(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!__has_unique_object_representations(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!__has_unique_object_representations(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));

#define ASSIGNABLE_WRAPPER(Type) __is_trivially_assignable(Type&, Type)
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedPtr) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedInt64) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(RelocatableAddressDiscriminatedFields) == !ADDR_DISC_ENABLED);
static_assert( ASSIGNABLE_WRAPPER(AddressDiscriminatedFieldInBaseClass) == !ADDR_DISC_ENABLED);
static_assert(!ASSIGNABLE_WRAPPER(NonAddressDiscriminatedVTablePtr));
static_assert(!ASSIGNABLE_WRAPPER(NonAddressDiscriminatedVTablePtr2));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableNonAddressDiscriminatedVTablePtr));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableNonAddressDiscriminatedVTablePtr2));
static_assert(!ASSIGNABLE_WRAPPER(AddressDiscriminatedVTablePtr));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableAddressDiscriminatedVTablePtr));
static_assert(!ASSIGNABLE_WRAPPER(NoAddressDiscriminatedBaseClasses));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableNoAddressDiscriminatedBaseClasses));
static_assert(!ASSIGNABLE_WRAPPER(AddressDiscriminatedPrimaryBase));
static_assert(!ASSIGNABLE_WRAPPER(AddressDiscriminatedSecondaryBase));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableAddressDiscriminatedPrimaryBase));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableAddressDiscriminatedSecondaryBase));
static_assert(!ASSIGNABLE_WRAPPER(EmbdeddedAddressDiscriminatedPolymorphicClass));
static_assert(!ASSIGNABLE_WRAPPER(RelocatableEmbdeddedAddressDiscriminatedPolymorphicClass));
