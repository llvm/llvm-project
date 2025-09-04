// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++26 -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++26 -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)
#define AQ_IP __ptrauth(1,1,50)
#define IQ_IP __ptrauth(1,0,50)
#define AA [[clang::ptrauth_vtable_pointer(process_independent,address_discrimination,no_extra_discrimination)]]
#define IA [[clang::ptrauth_vtable_pointer(process_independent,no_address_discrimination,type_discrimination)]]
#define PA [[clang::ptrauth_vtable_pointer(process_dependent,no_address_discrimination,no_extra_discrimination)]]

template <class T>
struct Holder {
  T t_;
  bool operator==(const Holder&) const = default;
};

struct S1 {
  int * AQ p_;
  void *payload_;
  bool operator==(const S1&) const = default;
};
static_assert(__is_trivially_constructible(S1));
static_assert(!__is_trivially_constructible(S1, const S1&));
static_assert(!__is_trivially_assignable(S1, const S1&));
static_assert(__is_trivially_destructible(S1));
static_assert(!__is_trivially_copyable(S1));
static_assert(!__is_trivially_relocatable(S1)); // expected-warning{{deprecated}}
static_assert(!__builtin_is_cpp_trivially_relocatable(S1));
static_assert(!__is_trivially_equality_comparable(S1));

static_assert(__is_trivially_constructible(Holder<S1>));
static_assert(!__is_trivially_constructible(Holder<S1>, const Holder<S1>&));
static_assert(!__is_trivially_assignable(Holder<S1>, const Holder<S1>&));
static_assert(__is_trivially_destructible(Holder<S1>));
static_assert(!__is_trivially_copyable(Holder<S1>));
static_assert(!__is_trivially_relocatable(Holder<S1>)); // expected-warning{{deprecated}}
static_assert(!__builtin_is_cpp_trivially_relocatable(Holder<S1>));
static_assert(!__is_trivially_equality_comparable(Holder<S1>));

struct S2 {
  int * IQ p_;
  void *payload_;
  bool operator==(const S2&) const = default;
};
static_assert(__is_trivially_constructible(S2));
static_assert(__is_trivially_constructible(S2, const S2&));
static_assert(__is_trivially_assignable(S2, const S2&));
static_assert(__is_trivially_destructible(S2));
static_assert(__is_trivially_copyable(S2));
static_assert(__is_trivially_relocatable(S2)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(S2));
static_assert(__is_trivially_equality_comparable(S2));

static_assert(__is_trivially_constructible(Holder<S2>));
static_assert(__is_trivially_constructible(Holder<S2>, const Holder<S2>&));
static_assert(__is_trivially_assignable(Holder<S2>, const Holder<S2>&));
static_assert(__is_trivially_destructible(Holder<S2>));
static_assert(__is_trivially_copyable(Holder<S2>));
static_assert(__is_trivially_relocatable(Holder<S2>)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(Holder<S2>));
static_assert(__is_trivially_equality_comparable(Holder<S2>));

struct AA S3 {
  virtual void f();
  void *payload_;
  bool operator==(const S3&) const = default;
};

static_assert(!__is_trivially_constructible(S3));
static_assert(!__is_trivially_constructible(S3, const S3&));
static_assert(!__is_trivially_assignable(S3, const S3&));
static_assert(__is_trivially_destructible(S3));
static_assert(!__is_trivially_copyable(S3));
static_assert(!__is_trivially_relocatable(S3)); // expected-warning{{deprecated}}
//FIXME
static_assert(!__builtin_is_cpp_trivially_relocatable(S3));
static_assert(!__is_trivially_equality_comparable(S3));


static_assert(!__is_trivially_constructible(Holder<S3>));
static_assert(!__is_trivially_constructible(Holder<S3>, const Holder<S3>&));
static_assert(!__is_trivially_assignable(Holder<S3>, const Holder<S3>&));
static_assert(__is_trivially_destructible(Holder<S3>));
static_assert(!__is_trivially_copyable(Holder<S3>));
static_assert(!__is_trivially_relocatable(Holder<S3>)); // expected-warning{{deprecated}}
static_assert(!__builtin_is_cpp_trivially_relocatable(Holder<S3>));
static_assert(!__is_trivially_equality_comparable(Holder<S3>));

struct IA S4 {
  virtual void f();
  void *payload_;
  bool operator==(const S4&) const = default;
};

static_assert(!__is_trivially_constructible(S4));
static_assert(!__is_trivially_constructible(S4, const S4&));
static_assert(!__is_trivially_assignable(S4, const S4&));
static_assert(__is_trivially_destructible(S4));
static_assert(!__is_trivially_copyable(S4));
static_assert(!__is_trivially_relocatable(S4)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(S4));
static_assert(!__is_trivially_equality_comparable(S4));

static_assert(!__is_trivially_constructible(Holder<S4>));
static_assert(!__is_trivially_constructible(Holder<S4>, const Holder<S4>&));
static_assert(!__is_trivially_assignable(Holder<S4>, const Holder<S4>&));
static_assert(__is_trivially_destructible(Holder<S4>));
static_assert(!__is_trivially_copyable(Holder<S4>));
static_assert(__is_trivially_relocatable(Holder<S4>)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(Holder<S4>));
static_assert(!__is_trivially_equality_comparable(Holder<S4>));

struct PA S5 {
  virtual void f();
  void *payload_;
  bool operator==(const S5&) const = default;
};

static_assert(!__is_trivially_constructible(S5));
static_assert(!__is_trivially_constructible(S5, const S5&));
static_assert(!__is_trivially_assignable(S5, const S5&));
static_assert(__is_trivially_destructible(S5));
static_assert(!__is_trivially_copyable(S5));
static_assert(!__is_trivially_relocatable(S5)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(S5));
static_assert(!__is_trivially_equality_comparable(S5));

static_assert(!__is_trivially_constructible(Holder<S5>));
static_assert(!__is_trivially_constructible(Holder<S5>, const Holder<S5>&));
static_assert(!__is_trivially_assignable(Holder<S5>, const Holder<S5>&));
static_assert(__is_trivially_destructible(Holder<S5>));
static_assert(!__is_trivially_copyable(Holder<S5>));
static_assert(__is_trivially_relocatable(Holder<S5>)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(Holder<S5>));
static_assert(!__is_trivially_equality_comparable(Holder<S5>));

struct S6 {
  __INTPTR_TYPE__ AQ_IP p_;
  void *payload_;
  bool operator==(const S6&) const = default;
};
static_assert(__is_trivially_constructible(S6));
static_assert(!__is_trivially_constructible(S6, const S6&));
static_assert(!__is_trivially_assignable(S6, const S6&));
static_assert(__is_trivially_destructible(S6));
static_assert(!__is_trivially_copyable(S6));
static_assert(!__is_trivially_relocatable(S6)); // expected-warning{{deprecated}}
static_assert(!__builtin_is_cpp_trivially_relocatable(S6));
static_assert(!__is_trivially_equality_comparable(S6));

static_assert(__is_trivially_constructible(Holder<S6>));
static_assert(!__is_trivially_constructible(Holder<S6>, const Holder<S6>&));
static_assert(!__is_trivially_assignable(Holder<S6>, const Holder<S6>&));
static_assert(__is_trivially_destructible(Holder<S6>));
static_assert(!__is_trivially_copyable(Holder<S6>));
static_assert(!__is_trivially_relocatable(Holder<S6>)); // expected-warning{{deprecated}}
static_assert(!__builtin_is_cpp_trivially_relocatable(Holder<S6>));
static_assert(!__is_trivially_equality_comparable(Holder<S6>));

struct S7 {
  __INTPTR_TYPE__ IQ_IP p_;
  void *payload_;
  bool operator==(const S7&) const = default;
};
static_assert(__is_trivially_constructible(S7));
static_assert(__is_trivially_constructible(S7, const S7&));
static_assert(__is_trivially_assignable(S7&, const S7&));
static_assert(__is_trivially_destructible(S7));
static_assert(__is_trivially_copyable(S7));
static_assert(__is_trivially_relocatable(S7)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(S7));
static_assert(__is_trivially_equality_comparable(S7));

static_assert(__is_trivially_constructible(Holder<S7>));
static_assert(__is_trivially_constructible(Holder<S7>, const Holder<S7>&));
static_assert(__is_trivially_assignable(Holder<S7>, const Holder<S7>&));
static_assert(__is_trivially_destructible(Holder<S7>));
static_assert(__is_trivially_copyable(Holder<S7>));
static_assert(__is_trivially_relocatable(Holder<S7>)); // expected-warning{{deprecated}}
static_assert(__builtin_is_cpp_trivially_relocatable(Holder<S7>));
static_assert(__is_trivially_equality_comparable(Holder<S7>));

template <class... Bases> struct MultipleInheriter : Bases... {
};

template <class T> static const bool test_is_trivially_relocatable_v = __builtin_is_cpp_trivially_relocatable(T);
template <class... Types> static const bool multiple_inheritance_is_relocatable = test_is_trivially_relocatable_v<MultipleInheriter<Types...>>;
template <class... Types> static const bool inheritance_relocatability_matches_bases_v =
  (test_is_trivially_relocatable_v<Types> && ...) == multiple_inheritance_is_relocatable<Types...>;

static_assert(multiple_inheritance_is_relocatable<S4, S5> == multiple_inheritance_is_relocatable<S5, S4>);
static_assert(inheritance_relocatability_matches_bases_v<S4, S5>);
static_assert(inheritance_relocatability_matches_bases_v<S5, S4>);

struct AA AddressDiscriminatedPolymorphicBase trivially_relocatable_if_eligible {
  virtual void foo();
};

struct IA NoAddressDiscriminatedPolymorphicBase trivially_relocatable_if_eligible {
  virtual void bar();
};

template <class T> struct UnionWrapper trivially_relocatable_if_eligible {
  union U {
    T field1;
  } u;
};

static_assert(!test_is_trivially_relocatable_v<AddressDiscriminatedPolymorphicBase>);
static_assert(test_is_trivially_relocatable_v<NoAddressDiscriminatedPolymorphicBase>);
static_assert(inheritance_relocatability_matches_bases_v<AddressDiscriminatedPolymorphicBase, NoAddressDiscriminatedPolymorphicBase>);
static_assert(inheritance_relocatability_matches_bases_v<NoAddressDiscriminatedPolymorphicBase, AddressDiscriminatedPolymorphicBase>);

static_assert(!test_is_trivially_relocatable_v<UnionWrapper<AddressDiscriminatedPolymorphicBase>>);
static_assert(test_is_trivially_relocatable_v<UnionWrapper<NoAddressDiscriminatedPolymorphicBase>>);
static_assert(!test_is_trivially_relocatable_v<UnionWrapper<MultipleInheriter<NoAddressDiscriminatedPolymorphicBase, AddressDiscriminatedPolymorphicBase>>>);
static_assert(!test_is_trivially_relocatable_v<UnionWrapper<MultipleInheriter<AddressDiscriminatedPolymorphicBase, NoAddressDiscriminatedPolymorphicBase>>>);
