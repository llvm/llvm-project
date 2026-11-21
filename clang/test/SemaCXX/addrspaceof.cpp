// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -fexperimental-new-constant-interpreter %s

#if !__has_extension(addrspaceof)
#error "missing addrspaceof extension"
#endif

using AS1 = int __attribute__((address_space(1)));
using AS2 = int __attribute__((address_space(2)));

static_assert(__addrspaceof(int) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(AS1) == __ADDRSPACE_TARGET(1));
static_assert(__addrspaceof(AS2 &) == __ADDRSPACE_TARGET(2));

int *p0;
AS1 *p1;

static_assert(__addrspaceof(p0) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(p1) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(*p0) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(*p1) == __ADDRSPACE_TARGET(1));

int global;
AS1 global_as1;
int array[4];
AS2 array_as2[4];

static_assert(__addrspaceof(global) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(global_as1) == __ADDRSPACE_TARGET(1));
static_assert(__addrspaceof(array) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(array_as2) == __ADDRSPACE_TARGET(2));
static_assert(__addrspaceof(array_as2[0]) == __ADDRSPACE_TARGET(2));

struct S {
  int member;
  static AS1 member_as1;
};
AS1 S::member_as1;

S object;
static_assert(__addrspaceof(object.member) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(object.member_as1) == __ADDRSPACE_TARGET(1));

int structured_array[3];
auto __attribute__((address_space(12))) [a, b, c] = structured_array;
static_assert(__addrspaceof(a) == __ADDRSPACE_TARGET(12));

static_assert(__addrspaceof((int [[clang::address_space(12)]]){100}) ==
              __ADDRSPACE_TARGET(12));
static_assert(__addrspaceof(0) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(&global) == __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(p0 + 1) == __ADDRSPACE_DEFAULT);

enum Kind { Item };
static_assert(__addrspaceof(Item) == __ADDRSPACE_DEFAULT);

template <int N> constexpr int nontype_parameter_address_space() {
  return __addrspaceof(N);
}
static_assert(nontype_parameter_address_space<3>() == __ADDRSPACE_DEFAULT);

template <class T> constexpr int type_address_space() {
  return __addrspaceof(T);
}

template <class T> constexpr int expression_address_space(T &value) {
  return __addrspaceof(value);
}

template <class T> constexpr int prvalue_address_space(T *value) {
  return __addrspaceof(value + 1);
}

static_assert(type_address_space<AS1>() == __ADDRSPACE_TARGET(1));
static_assert(expression_address_space(global_as1) == __ADDRSPACE_TARGET(1));
static_assert(prvalue_address_space(&global_as1) == __ADDRSPACE_DEFAULT);

template <int AS> struct AddressSpaceSpecialization;
template <>
struct AddressSpaceSpecialization<__ADDRSPACE_TARGET(1)> {
  static constexpr int value = __ADDRSPACE_TARGET(1);
};

static_assert(AddressSpaceSpecialization<__addrspaceof(AS1)>::value ==
              __ADDRSPACE_TARGET(1));

void function();

void errors() {
  (void)__addrspaceof global;
  // expected-error@-1 {{expected '(' after '__addrspaceof'}}
  (void)__addrspaceof(function);
  // expected-error@-1 {{unparenthesized function name is not a valid operand of '__addrspaceof'}}
}
