// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -fexperimental-new-constant-interpreter %s

#if !__has_extension(addrspaceof)
#error "missing addrspaceof extension"
#endif

using AS1 = int __attribute__((address_space(1)));
using AS2 = int __attribute__((address_space(2)));

static_assert(__addrspaceof(int) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(AS1) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);
static_assert(__addrspaceof(AS2 &) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 2);

int *p0;
AS1 *p1;

static_assert(__addrspaceof(p0) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(p1) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(*p0) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(*p1) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);

int global;
AS1 global_as1;
int array[4];
AS2 array_as2[4];

static_assert(__addrspaceof(global) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(global_as1) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);
static_assert(__addrspaceof(array) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(array_as2) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 2);
static_assert(__addrspaceof(array_as2[0]) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 2);

struct S {
  int member;
  static AS1 member_as1;
};
AS1 S::member_as1;

S object;
static_assert(__addrspaceof(object.member) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(object.member_as1) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);

template <class T> constexpr int type_address_space() {
  return __addrspaceof(T);
}

template <class T> constexpr int expression_address_space(T &value) {
  return __addrspaceof(value);
}

static_assert(type_address_space<AS1>() ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);
static_assert(expression_address_space(global_as1) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);

template <int AS> struct AddressSpaceSpecialization;
template <>
struct AddressSpaceSpecialization<__CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1> {
  static constexpr int value = __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1;
};

static_assert(AddressSpaceSpecialization<__addrspaceof(AS1)>::value ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);

void function();

void errors() {
  (void)__addrspaceof global;
  // expected-error@-1 {{expected '(' after '__addrspaceof'}}
  (void)__addrspaceof(0);
  // expected-error@-1 {{expression operand of '__addrspaceof' must be an lvalue}}
  (void)__addrspaceof(&global);
  // expected-error@-1 {{expression operand of '__addrspaceof' must be an lvalue}}
  (void)__addrspaceof(p0 + 1);
  // expected-error@-1 {{expression operand of '__addrspaceof' must be an lvalue}}
  (void)__addrspaceof(function);
  // expected-error@-1 {{entity operand of '__addrspaceof' must name a variable or data member}}
}
