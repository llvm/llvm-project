// RUN: %clang_cc1 -std=c++20 -Wno-unused %s -verify=expected,cxx20
// RUN: %clang_cc1 -std=c++23 -Wno-unused %s -verify=expected
// RUN: %clang_cc1 -std=c++2c -Wno-unused %s -verify=expected

inline void* operator new(__SIZE_TYPE__, void* p) noexcept { return p; }
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

consteval bool test_dynamic() {
  std::allocator<int> a;
  int* p = a.allocate(1);
  if (__builtin_is_within_lifetime(p))
    return false;
  std::construct_at(p);
  if (!__builtin_is_within_lifetime(p))
    return false;
  std::destroy_at(p);
  if (__builtin_is_within_lifetime(p))
    return false;
  a.deallocate(p, 1);
  if (__builtin_is_within_lifetime(p))
    return false;
  return true;
}
static_assert(test_dynamic());

consteval bool test_automatic() {
  int* p;
  {
    int x = 0;
    p = &x;
    if (!__builtin_is_within_lifetime(p))
      return false;
  }
  {
    int x = 0;
    if (__builtin_is_within_lifetime(p))
      return false;
  }
  if (__builtin_is_within_lifetime(p))
    return false;
  {
    int x[4];
    p = &x[2];
    if (!__builtin_is_within_lifetime(p))
      return false;
  }
  if (__builtin_is_within_lifetime(p))
    return false;
  std::nullptr_t* q;
  {
    std::nullptr_t np = nullptr;
    q = &np;
    if (!__builtin_is_within_lifetime(q))
      return false;
  }
  if (__builtin_is_within_lifetime(q))
    return false;
  return true;
}
static_assert(test_automatic());

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
constexpr int external{};
static_assert(__builtin_is_within_lifetime(&external));
void not_constexpr() {
  __builtin_is_within_lifetime(&external);
}
void invalid_args() {
  __builtin_is_within_lifetime(static_cast<int*>(nullptr));
// expected-error@-1 {{'__builtin_is_within_lifetime' cannot be called with a null pointer}}
// expected-error@-2 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
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

constexpr struct {
  bool a = __builtin_is_within_lifetime(&b);
  bool b = __builtin_is_within_lifetime(&a);
  bool c = __builtin_is_within_lifetime(this);
} x2;
static_assert(!x2.a);
static_assert(!x2.b);
static_assert(!x2.c);

struct X3 {
  bool a, b, c, d;
  consteval X3();
};
extern const X3 x3;
consteval X3::X3() : a(__builtin_is_within_lifetime(&b)), b(false), c(__builtin_is_within_lifetime(&b)) {
  b = __builtin_is_within_lifetime(&b);
  d = __builtin_is_within_lifetime(&x3.c);
}
constexpr X3 x3{};
static_assert(!x3.a);
static_assert(!x3.b);
static_assert(!x3.c);
static_assert(!x3.d);

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
// expected-error@-2 {{'__builtin_is_within_lifetime' cannot be called with a one-past-the-end pointer}}
  __builtin_is_within_lifetime(arr + 2);
// expected-error@-1 {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}}
// expected-error@-2 {{'__builtin_is_within_lifetime' cannot be called with a one-past-the-end pointer}}
}

template<typename T>
consteval bool allow_bad_types_unless_used(bool b, T* p) {
  if (b) {
    __builtin_is_within_lifetime(p); // #bad_type_used
  }
  return true;
}
void fn();
static_assert(allow_bad_types_unless_used<void()>(false, &fn));
void g() {
  allow_bad_types_unless_used<void()>(true, &fn);
// expected-error@-1 {{call to consteval function 'allow_bad_types_unless_used<void ()>' is not a constant expression}}
// expected-error@#bad_type_used {{'__builtin_is_within_lifetime' cannot be called with a function pointer}}
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
