// RUN: %clang_cc1 -std=c++20 -Wno-unused %s -verify=expected,cxx20 -Wno-vla-cxx-extension
// RUN: %clang_cc1 -std=c++23 -Wno-unused %s -verify=expected,sincecxx23 -Wno-vla-cxx-extension
// RUN: %clang_cc1 -std=c++26 -Wno-unused %s -verify=expected,sincecxx23 -Wno-vla-cxx-extension
// RUN: %clang_cc1 -std=c++26 -DINLINE_NAMESPACE -Wno-unused %s -verify=expected,sincecxx23 -Wno-vla-cxx-extension

inline constexpr void* operator new(__SIZE_TYPE__, void* p) noexcept { return p; }
namespace std {
template<typename T, typename... Args>
constexpr T* construct_at(T* p, Args&&... args) { return ::new((void*)p) T(static_cast<Args&&>(args)...); }
template<typename T>
constexpr void destroy_at(T* p) { p->~T(); }
template<typename T>
struct allocator {
  constexpr T* allocate(__SIZE_TYPE__ n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
  constexpr void deallocate(T* p, __SIZE_TYPE__) { ::operator delete(p); }
};
using nullptr_t = decltype(nullptr);
template<typename T, T v>
struct integral_constant { static constexpr T value = v; };
template<bool v>
using bool_constant = integral_constant<bool, v>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;
template<typename T>
inline constexpr bool is_function_v = __is_function(T);
#ifdef INLINE_NAMESPACE
inline namespace __1 {
#endif
template<typename T> requires (!is_function_v<T>) // #std-constraint
consteval bool is_within_lifetime(const T* p) noexcept { // #std-definition
  return __builtin_is_within_lifetime(p);
}
#ifdef INLINE_NAMESPACE
}
#endif
}

consteval bool test_union(int& i, char& c) {
  if (__builtin_is_within_lifetime(&i) || __builtin_is_within_lifetime(&c))
    return false;
  std::construct_at(&c, 1);
  if (__builtin_is_within_lifetime(&i) || !__builtin_is_within_lifetime(&c))
    return false;
  std::construct_at(&i, 3);
  if (!__builtin_is_within_lifetime(&i) || __builtin_is_within_lifetime(&c))
    return false;
  return true;
}

static_assert([]{
  union { int i; char c; } u;
  return test_union(u.i, u.c);
}());
static_assert([]{
  union { int i; char c; };
  return test_union(i, c);
}());
static_assert([]{
  struct { union { int i; char c; }; } u;
  return test_union(u.i, u.c);
}());
static_assert([]{
  struct { union { int i; char c; } u; } r;
  return test_union(r.u.i, r.u.c);
}());

consteval bool test_nested() {
  union {
    union { int i; char c; } u;
    long l;
  };
  if (__builtin_is_within_lifetime(&l) || __builtin_is_within_lifetime(&u) || __builtin_is_within_lifetime(&u.i) || __builtin_is_within_lifetime(&u.c))
    return false;
  std::construct_at(&l);
  if (!__builtin_is_within_lifetime(&l) || __builtin_is_within_lifetime(&u) || __builtin_is_within_lifetime(&u.i) || __builtin_is_within_lifetime(&u.c))
    return false;
  std::construct_at(&u);
  std::construct_at(&u.i);
  if (__builtin_is_within_lifetime(&l) || !__builtin_is_within_lifetime(&u) || !__builtin_is_within_lifetime(&u.i) || __builtin_is_within_lifetime(&u.c))
    return false;
  std::construct_at(&u.c);
  if (__builtin_is_within_lifetime(&l) || !__builtin_is_within_lifetime(&u) || __builtin_is_within_lifetime(&u.i) || !__builtin_is_within_lifetime(&u.c))
    return false;
  return true;
}
static_assert(test_nested());

consteval bool test_dynamic(bool read_after_deallocate) {
  std::allocator<int> a;
  int* p = a.allocate(1);
  // a.allocate starts the lifetime of an array,
  // the complete object of *p has started its lifetime
  if (__builtin_is_within_lifetime(p))
    return false;
  std::construct_at(p);
  if (!__builtin_is_within_lifetime(p))
    return false;
  std::destroy_at(p);
  if (__builtin_is_within_lifetime(p))
    return false;
  a.deallocate(p, 1);
  if (read_after_deallocate)
    __builtin_is_within_lifetime(p); // expected-note {{read of heap allocated object that has been deleted}}
  return true;
}
static_assert(test_dynamic(false));
static_assert(test_dynamic(true));
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{in call to 'test_dynamic(true)'}}

consteval bool test_automatic(int read_dangling) {
  int* p;
  {
    int x = 0;
    p = &x;
    if (!__builtin_is_within_lifetime(p))
      return false;
  }
  {
    int x = 0;
    if (read_dangling == 1)
      __builtin_is_within_lifetime(p); // expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
  }
  if (read_dangling == 2)
    __builtin_is_within_lifetime(p); // expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
  {
    int x[4];
    p = &x[2];
    if (!__builtin_is_within_lifetime(p))
      return false;
  }
  if (read_dangling == 3)
    __builtin_is_within_lifetime(p); // expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
  std::nullptr_t* q;
  {
    std::nullptr_t np = nullptr;
    q = &np;
    if (!__builtin_is_within_lifetime(q))
      return false;
  }
  if (read_dangling == 4)
    __builtin_is_within_lifetime(q); // expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
  return true;
}
static_assert(test_automatic(0));
static_assert(test_automatic(1));
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{in call to 'test_automatic(1)'}}
static_assert(test_automatic(2));
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{in call to 'test_automatic(2)'}}
static_assert(test_automatic(3));
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{in call to 'test_automatic(3)'}}
static_assert(test_automatic(4));
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{in call to 'test_automatic(4)'}}


consteval bool test_indeterminate() {
  int x;
  if (!__builtin_is_within_lifetime(&x))
    return false;
  bool b = true;
  unsigned char c = __builtin_bit_cast(unsigned char, b);
  if (!__builtin_is_within_lifetime(&c))
    return false;
  struct {} padding;
  unsigned char y = __builtin_bit_cast(unsigned char, padding);
  if (!__builtin_is_within_lifetime(&y))
    return false;
  return true;
}
static_assert(test_indeterminate());

consteval bool test_volatile() {
  int x;
  if (!__builtin_is_within_lifetime(static_cast<volatile int*>(&x)) || !__builtin_is_within_lifetime(static_cast<volatile void*>(&x)))
    return false;
  volatile int y;
  if (!__builtin_is_within_lifetime(const_cast<int*>(&y)) || !__builtin_is_within_lifetime(const_cast<void*>(static_cast<volatile void*>(&y))))
    return false;
  return true;
}
static_assert(test_volatile());

constexpr bool self = __builtin_is_within_lifetime(&self);
// expected-error@-1 {{constexpr variable 'self' must be initialized by a constant expression}}
//   expected-note@-2 {{'__builtin_is_within_lifetime' cannot be called with a pointer to an object whose lifetime has not yet begun}}
// expected-error@-3 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
//   expected-note@-4 {{initializer of 'self' is not a constant expression}}
//   expected-note@-5 {{declared here}}
constexpr int external{};
static_assert(__builtin_is_within_lifetime(&external));
void not_constexpr() {
  __builtin_is_within_lifetime(&external);
}
void invalid_args() {
  __builtin_is_within_lifetime(static_cast<int*>(nullptr));
  // expected-error@-1 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
  //   expected-note@-2 {{'__builtin_is_within_lifetime' cannot be called with a null pointer}}

  // FIXME: avoid function to pointer conversion on all consteval builtins
  __builtin_is_within_lifetime(0);
  // expected-error@-1 {{non-pointer argument to '__builtin_is_within_lifetime' is not allowed}}
  // expected-error@-2 {{cannot take address of consteval function '__builtin_is_within_lifetime' outside of an immediate invocation}}
  __builtin_is_within_lifetime();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // expected-error@-2 {{cannot take address of consteval function '__builtin_is_within_lifetime' outside of an immediate invocation}}
  __builtin_is_within_lifetime(1, 2);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  // expected-error@-2 {{cannot take address of consteval function '__builtin_is_within_lifetime' outside of an immediate invocation}}
  __builtin_is_within_lifetime(&external, &external);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  // expected-error@-2 {{cannot take address of consteval function '__builtin_is_within_lifetime' outside of an immediate invocation}}
}

constexpr struct {
  union {
    int i;
    char c;
  };
  mutable int mi;  // #x-mi
} x1{ .c = 2 };
static_assert(!__builtin_is_within_lifetime(&x1.i));
static_assert(__builtin_is_within_lifetime(&x1.c));
static_assert(__builtin_is_within_lifetime(&x1.mi));
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{read of mutable member 'mi' is not allowed in a constant expression}}
//   expected-note@#x-mi {{declared here}}

constexpr struct NSDMI { // #NSDMI
  bool a = true;
  bool b = __builtin_is_within_lifetime(&a); // #NSDMI-read
} x2;
// expected-error@-1 {{constexpr variable 'x2' must be initialized by a constant expression}}
//   expected-note@#NSDMI-read {{'__builtin_is_within_lifetime' cannot be called with a pointer to an object whose lifetime has not yet begun}}
//   expected-note@-3 {{in call to 'NSDMI()'}}
// expected-error@-4 {{call to immediate function 'NSDMI::NSDMI' is not a constant expression}}
//   expected-note@#NSDMI {{'NSDMI' is an immediate constructor because the default initializer of 'b' contains a call to a consteval function '__builtin_is_within_lifetime' and that call is not a constant expression}}
//   expected-note@#NSDMI-read {{'__builtin_is_within_lifetime' cannot be called with a pointer to an object whose lifetime has not yet begun}}
//   expected-note@-7 {{in call to 'NSDMI()'}}

struct X3 {
  consteval X3() {
    __builtin_is_within_lifetime(this); // #X3-read
  }
} x3;
// expected-error@-1 {{call to consteval function 'X3::X3' is not a constant expression}}
//   expected-note@#X3-read {{'__builtin_is_within_lifetime' cannot be called with a pointer to an object whose lifetime has not yet begun}}
//   expected-note@-3 {{in call to 'X3()'}}

constexpr int i = 2;
static_assert(__builtin_is_within_lifetime(const_cast<int*>(&i)));
static_assert(__builtin_is_within_lifetime(const_cast<volatile int*>(&i)));
static_assert(__builtin_is_within_lifetime(static_cast<const void*>(&i)));

constexpr int arr[2]{};
static_assert(__builtin_is_within_lifetime(arr));
static_assert(__builtin_is_within_lifetime(arr + 0));
static_assert(__builtin_is_within_lifetime(arr + 1));
void f() {
  __builtin_is_within_lifetime(&i + 1);
  // expected-error@-1 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
  //   expected-note@-2 {{'__builtin_is_within_lifetime' cannot be called with a one-past-the-end pointer}}
  __builtin_is_within_lifetime(arr + 2);
  // expected-error@-1 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
  //   expected-note@-2 {{'__builtin_is_within_lifetime' cannot be called with a one-past-the-end pointer}}
}

template<typename T>
consteval void disallow_function_types(bool b, const T* p) {
  if (b) {
    __builtin_is_within_lifetime(p); // expected-error {{function pointer argument to '__builtin_is_within_lifetime' is not allowed}}
  }
}
void g() {
  disallow_function_types<void ()>(false, &f);
  // expected-note@-1 {{in instantiation of function template specialization 'disallow_function_types<void ()>' requested here}}
}

struct OptBool {
  union { bool b; char c; };

  // note: this assumes common implementation properties for bool and char:
  // * sizeof(bool) == sizeof(char), and
  // * the value representations for true and false are distinct
  //   from the value representation for 2
  constexpr OptBool() : c(2) { }
  constexpr OptBool(bool b) : b(b) { }

  constexpr auto has_value() const -> bool {
    if consteval {  // cxx20-warning {{consteval if}}
      return __builtin_is_within_lifetime(&b);   // during constant evaluation, cannot read from c
    } else {
      return c != 2;                        // during runtime, must read from c
    }
  }

  constexpr auto operator*() const -> const bool& {
    return b;
  }
};

constexpr OptBool disengaged;
constexpr OptBool engaged(true);
static_assert(!disengaged.has_value());
static_assert(engaged.has_value());
static_assert(*engaged);

namespace vlas {

consteval bool f(int n) {
  int vla[n]; // cxx20-error {{variable of non-literal type}}
  return __builtin_is_within_lifetime(static_cast<void*>(&vla));
}
static_assert(f(1));

consteval bool fail(int n) {
  int vla[n]; // cxx20-error {{variable of non-literal type}}
  return __builtin_is_within_lifetime(&vla); // expected-error {{variable length arrays are not supported in '__builtin_is_within_lifetime'}}
}
static_assert(fail(1)); // sincecxx23-error {{static assertion expression is not an integral constant expression}}

consteval bool variably_modified(int n) {
  int(* p)[n];
  return __builtin_is_within_lifetime(&p);
}
static_assert(variably_modified(1));

} // namespace vlas

consteval bool partial_arrays() {
  int arr[2];
  if (!__builtin_is_within_lifetime(&arr) || !__builtin_is_within_lifetime(&arr[0]) || !__builtin_is_within_lifetime(&arr[1]))
    return false;
  std::destroy_at(&arr[0]);
  if (!__builtin_is_within_lifetime(&arr) ||  __builtin_is_within_lifetime(&arr[0]) || !__builtin_is_within_lifetime(&arr[1]))
    return false;
  std::construct_at(&arr[0]);
  if (!__builtin_is_within_lifetime(&arr) || !__builtin_is_within_lifetime(&arr[0]) || !__builtin_is_within_lifetime(&arr[1]))
    return false;
  return true;
}
static_assert(partial_arrays());

consteval bool partial_members() {
  struct S {
    int x;
    int y;
  } s;
  if (!__builtin_is_within_lifetime(&s) || !__builtin_is_within_lifetime(&s.x) || !__builtin_is_within_lifetime(&s.y))
    return false;
  std::destroy_at(&s.x);
  if (!__builtin_is_within_lifetime(&s) ||  __builtin_is_within_lifetime(&s.x) || !__builtin_is_within_lifetime(&s.y))
    return false;
  std::construct_at(&s.x);
  if (!__builtin_is_within_lifetime(&s) || !__builtin_is_within_lifetime(&s.x) || !__builtin_is_within_lifetime(&s.y))
    return false;
  return true;
}

struct NonTrivial {
  constexpr NonTrivial() {}
  constexpr NonTrivial(const NonTrivial&) {}
  constexpr ~NonTrivial() {}
};

template<typename T>
constexpr T& unmove(T&& temp) { return static_cast<T&>(temp); }

consteval bool test_temporaries() {
  static_assert(__builtin_is_within_lifetime(&unmove(0)));
  static_assert(__builtin_is_within_lifetime(&unmove(NonTrivial{})));
  if (!__builtin_is_within_lifetime(&unmove(0)))
    return false;
  if (!__builtin_is_within_lifetime(&unmove(NonTrivial{})))
    return false;
  return true;
}
static_assert(test_temporaries());

constexpr const int& temp = 0;
static_assert(__builtin_is_within_lifetime(&temp));

template<typename T>
constexpr T* test_dangling() {
  T i; // expected-note 2 {{declared here}}
  return &i; // expected-warning 2 {{address of stack memory associated with local variable 'i' returned}}
}
static_assert(__builtin_is_within_lifetime(test_dangling<int>())); // expected-note {{in instantiation of function template specialization}}
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{read of variable whose lifetime has ended}}
static_assert(__builtin_is_within_lifetime(test_dangling<int[1]>())); // expected-note {{in instantiation of function template specialization}}
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{read of variable whose lifetime has ended}}

template<auto F>
concept CanCallAndPassToIsWithinLifetime = std::bool_constant<__builtin_is_within_lifetime(F())>::value;
static_assert(CanCallAndPassToIsWithinLifetime<[]{ return &i; }>);
static_assert(!CanCallAndPassToIsWithinLifetime<[]{ return static_cast<int*>(nullptr); }>);
static_assert(!CanCallAndPassToIsWithinLifetime<[]{ return static_cast<void(*)()>(&f); }>);
template<auto F> constexpr std::true_type sfinae() requires CanCallAndPassToIsWithinLifetime<F> { return {}; }
template<auto F> std::false_type sfinae() { return {}; }
static_assert(decltype(sfinae<[]{ return &i; }>())::value);
static_assert(!decltype(sfinae<[]{ return static_cast<int*>(nullptr); }>())::value);
std::true_type(* not_immediate)() = &sfinae<[]{ return &i; }>;

void test_std_error_message() {
  std::is_within_lifetime(static_cast<int*>(nullptr));
  // expected-error@-1 {{call to consteval function 'std::is_within_lifetime<int>' is not a constant expression}}
  //   expected-note@-2 {{'std::is_within_lifetime' cannot be called with a null pointer}}
  //   expected-note@-3 {{in call to 'is_within_lifetime<int>(nullptr)'}}
  std::is_within_lifetime<void()>(&test_std_error_message);
  // expected-error@-1 {{no matching function for call to 'is_within_lifetime'}}
  //   expected-note@#std-definition {{candidate template ignored: constraints not satisfied [with T = void ()]}}
  //   expected-note@#std-constraint {{because '!is_function_v<void ()>' evaluated to false}}
  std::is_within_lifetime(arr + 2);
  // expected-error@-1 {{call to consteval function 'std::is_within_lifetime<int>' is not a constant expression}}
  //   expected-note@-2 {{'std::is_within_lifetime' cannot be called with a one-past-the-end pointer}}
  //   expected-note@-3 {{in call to 'is_within_lifetime<int>(&arr[2])'}}
}
struct XStd {
  consteval XStd() {
    std::is_within_lifetime(this); // #XStd-read
  }
} xstd;
// expected-error@-1 {{call to consteval function 'XStd::XStd' is not a constant expression}}
//   expected-note@#XStd-read {{'std::is_within_lifetime' cannot be called with a pointer to an object whose lifetime has not yet begun}}
//   expected-note@#XStd-read {{in call to 'is_within_lifetime<XStd>(&)'}}
//   expected-note@-4 {{in call to 'XStd()'}}
