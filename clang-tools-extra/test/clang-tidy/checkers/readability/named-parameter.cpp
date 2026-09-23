// RUN: %check_clang_tidy %s readability-named-parameter %t
// RUN: %check_clang_tidy -check-suffix=PLAIN-NAMES %s readability-named-parameter %t -- \
// RUN:   -config="{CheckOptions: [{key: readability-named-parameter.InsertPlainNamesInForwardDecls, value: true}]}"

void Method(char *) { /* */ }
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: all parameters should be named in a function
// CHECK-FIXES: void Method(char * /*unused*/) { /* */ }
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:19: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void Method(char * /*unused*/) { /* */ }
void Method2(char *) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: all parameters should be named in a function
// CHECK-FIXES: void Method2(char * /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:20: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void Method2(char * /*unused*/) {}
void Method3(char *, void *) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: all parameters should be named in a function
// CHECK-FIXES: void Method3(char * /*unused*/, void * /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:20: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void Method3(char * /*unused*/, void * /*unused*/) {}
void Method4(char *, int /*unused*/) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: all parameters should be named in a function
// CHECK-FIXES: void Method4(char * /*unused*/, int /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:20: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void Method4(char * /*unused*/, int /*unused*/) {}
void operator delete[](void *) throw() {}
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: all parameters should be named in a function
// CHECK-FIXES: void operator delete[](void * /*unused*/) throw() {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:30: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void operator delete[](void * /*unused*/) throw() {}
int Method5(int) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: all parameters should be named in a function
// CHECK-FIXES: int Method5(int /*unused*/) { return 0; }
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:16: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: int Method5(int /*unused*/) { return 0; }
void Method6(void (*)(void *)) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: all parameters should be named in a function
// CHECK-FIXES: void Method6(void (* /*unused*/)(void *)) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:21: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void Method6(void (* /*unused*/)(void *)) {}
template <typename T> void Method7(T) {}
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: all parameters should be named in a function
// CHECK-FIXES: template <typename T> void Method7(T /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:37: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: template <typename T> void Method7(T /*unused*/) {}

// Don't warn in macros.
#define M void MethodM(int) {}
M

void operator delete(void *x) throw() {}
void Method7(char * /*x*/) {}
void Method8(char *x) {}
typedef void (*TypeM)(int x);
void operator delete[](void *x) throw();
void operator delete[](void * /*x*/) throw();

struct X {
  void operator++(int) {}
  void operator--(int) {}

  X(X&) = delete;
  X &operator=(X&) = default;

  const int &i;
};

void (*Func1)(void *);
void Func2(void (*func)(void *)) {}
template <void Func(void *)> void Func3() {}

template <typename T>
struct Y {
  void foo(T) {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: all parameters should be named in a function
// CHECK-FIXES: void foo(T /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:13: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void foo(T /*unused*/) {}
};

Y<int> y;
Y<float> z;

struct Base {
  virtual void foo(bool notThisOne);
  virtual void foo(int argname);
};

struct Derived : public Base {
  void foo(int);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: all parameters should be named in a function
// CHECK-FIXES: void foo(int /*argname*/);
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:15: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void foo(int argname);
};

void FDef(int);
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: all parameters should be named in a function
// CHECK-FIXES: void FDef(int /*n*/);
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:14: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void FDef(int n);
void FDef(int n) {}

void FDef2(int, int);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: all parameters should be named in a function
// CHECK-FIXES: void FDef2(int /*n*/, int /*unused*/);
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:15: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void FDef2(int n, int /*unused*/);
void FDef2(int n, int) {}
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: all parameters should be named in a function
// CHECK-FIXES: void FDef2(int n, int /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:22: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void FDef2(int n, int /*unused*/) {}

void FNoDef(int);

class Z {};
Z the_z;

Z &operator++(Z&) { return the_z; }
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: all parameters should be named in a function
// CHECK-FIXES: Z &operator++(Z& /*unused*/) { return the_z; }
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:17: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: Z &operator++(Z& /*unused*/) { return the_z; }

Z &operator++(Z&, int) { return the_z; }
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: all parameters should be named in a function
// CHECK-FIXES: Z &operator++(Z& /*unused*/, int) { return the_z; }
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:17: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: Z &operator++(Z& /*unused*/, int) { return the_z; }

Z &operator--(Z&) { return the_z; }
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: all parameters should be named in a function
// CHECK-FIXES: Z &operator--(Z& /*unused*/) { return the_z; }
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:17: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: Z &operator--(Z& /*unused*/) { return the_z; }

Z &operator--(Z&, int) { return the_z; }
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: all parameters should be named in a function
// CHECK-FIXES: Z &operator--(Z& /*unused*/, int) { return the_z; }
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:17: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: Z &operator--(Z& /*unused*/, int) { return the_z; }

namespace testing {
namespace internal {
class IgnoredValue {
 public:
  template <typename T>
  IgnoredValue(const T& /* ignored */) {}
};
}
typedef internal::IgnoredValue Unused;
}

using ::testing::Unused;

void MockFunction(Unused, int q, Unused) {
  ++q;
  ++q;
  ++q;
}

namespace std {
typedef decltype(nullptr) nullptr_t;
struct adopt_lock_t {};
struct allocator_arg_t {};
struct bidirectional_iterator_tag {};
struct contiguous_iterator_tag {};
struct default_sentinel_t {};
struct defer_lock_t {};
struct destroying_delete_t {};
struct forward_iterator_tag {};
struct from_range_t {};
template<unsigned I> struct in_place_index_t {};
struct in_place_t {};
template<class T> struct in_place_type_t {};
struct input_iterator_tag {};
struct nothrow_t {};
struct nostopstate_t {};
struct nullopt_t {};
struct output_iterator_tag {};
struct piecewise_construct_t {};
struct random_access_iterator_tag {};
struct sorted_equivalent_t {};
struct sorted_unique_t {};
struct try_to_lock_t {};
struct unexpect_t {};
struct unreachable_sentinel_t {};
}

void f(std::nullptr_t) {}

// Standard tag dispatch types should not trigger warnings.
void f_adopt_lock(std::adopt_lock_t) {}
void f_allocator_arg(std::allocator_arg_t) {}
void f_bidir_iter(std::bidirectional_iterator_tag) {}
void f_contig_iter(std::contiguous_iterator_tag) {}
void f_default_sentinel(std::default_sentinel_t) {}
void f_defer_lock(std::defer_lock_t) {}
void f_destroying(std::destroying_delete_t) {}
void f_forward_iter(std::forward_iterator_tag) {}
void f_from_range(std::from_range_t) {}
void f_in_place_index(std::in_place_index_t<0>) {}
void f_in_place(std::in_place_t) {}
void f_in_place_type(std::in_place_type_t<int>) {}
void f_input_iter(std::input_iterator_tag) {}
void f_nothrow(std::nothrow_t) {}
void f_nostopstate(std::nostopstate_t) {}
void f_nullopt(std::nullopt_t) {}
void f_output_iter(std::output_iterator_tag) {}
void f_piecewise(std::piecewise_construct_t) {}
void f_random_iter(std::random_access_iterator_tag) {}
void f_sorted_equivalent(std::sorted_equivalent_t) {}
void f_sorted_unique(std::sorted_unique_t) {}
void f_try_to_lock(std::try_to_lock_t) {}
void f_unexpect(std::unexpect_t) {}
void f_unreachable_sentinel(std::unreachable_sentinel_t) {}

// A type not in the IgnoredTypes list should still trigger a warning.
struct NotATag {};
void f_not_a_tag(NotATag) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: all parameters should be named in a function
// CHECK-FIXES: void f_not_a_tag(NotATag /*unused*/) {}
// CHECK-MESSAGES-PLAIN-NAMES: :[[@LINE-3]]:25: warning: all parameters should be named in a function
// CHECK-FIXES-PLAIN-NAMES: void f_not_a_tag(NotATag /*unused*/) {}

typedef void (F)(int);
F f;
void f(int x) {}

namespace issue_63056
{
  struct S {
    S(const S&);
    S(S&&);

    S& operator=(const S&);
    S& operator=(S&&);
  };

  S::S(const S&) = default;
  S::S(S&&) = default;

  S& S::operator=(const S&) = default;
  S& S::operator=(S&&) = default;
} // namespace issue_63056
