// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

#if !__has_builtin(__builtin_pointee_address_space)
#error "missing __builtin_pointee_address_space"
#endif

int *p0;
int __attribute__((address_space(1))) *p1;
const int __attribute__((address_space(7))) *p7;

static_assert(__builtin_pointee_address_space(p0) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__builtin_pointee_address_space(p1) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);
static_assert(__builtin_pointee_address_space(p7) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 7);
static_assert(__builtin_pointee_address_space(
                  (int __attribute__((address_space(3))) *)0) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 3);

int global;
int __attribute__((address_space(4))) global_as4;
int arr[4];
int __attribute__((address_space(5))) arr_as5[4];

static_assert(__builtin_pointee_address_space(&global) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__builtin_pointee_address_space(&global_as4) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 4);
static_assert(__builtin_pointee_address_space(arr) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__builtin_pointee_address_space(arr_as5) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 5);

template <class T> constexpr int get_as(T *p) {
  return __builtin_pointee_address_space(p);
}

static_assert(get_as((int *)0) == __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(get_as((int __attribute__((address_space(1))) *)0) ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);

template <class T> struct PointeeAddressSpace {
  static constexpr int value =
      __builtin_pointee_address_space((T *)0);
};

static_assert(PointeeAddressSpace<int>::value ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(PointeeAddressSpace<
              int __attribute__((address_space(2)))>::value ==
              __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 2);

template <int AS> struct AddressSpaceSpecialization;
template <>
struct AddressSpaceSpecialization<__CLANG_ADDRESS_SPACE_DEFAULT> {
  static constexpr int value = __CLANG_ADDRESS_SPACE_DEFAULT;
};
template <>
struct AddressSpaceSpecialization<__CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1> {
  static constexpr int value = __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1;
};
template <>
struct AddressSpaceSpecialization<__CLANG_ADDRESS_SPACE_TARGET_OFFSET + 5> {
  static constexpr int value = __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 5;
};

static_assert(
    AddressSpaceSpecialization<
        __builtin_pointee_address_space(p1)>::value ==
    __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 1);
static_assert(
    AddressSpaceSpecialization<
        __builtin_pointee_address_space(arr_as5)>::value ==
    __CLANG_ADDRESS_SPACE_TARGET_OFFSET + 5);

void errors() {
  int i;
  void f();

  (void)__builtin_pointee_address_space();
  // expected-error@-1 {{too few arguments}}
  (void)__builtin_pointee_address_space(p0, p1);
  // expected-error@-1 {{too many arguments}}
  (void)__builtin_pointee_address_space(i);
  // expected-error@-1 {{argument to '__builtin_pointee_address_space' must be a pointer or array expression}}
  (void)__builtin_pointee_address_space(0);
  // expected-error@-1 {{argument to '__builtin_pointee_address_space' must be a pointer or array expression}}
  (void)__builtin_pointee_address_space(f);
  // expected-error@-1 {{argument to '__builtin_pointee_address_space' must be a pointer or array expression}}
}
