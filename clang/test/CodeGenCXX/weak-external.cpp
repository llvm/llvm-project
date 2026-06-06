// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s
// PR4262

// CHECK-NOT: _ZNSs12_S_constructIPKcEEPcT_S3_RKSaIcESt20forward_iterator_tag

// The "basic_string" extern template instantiation declaration is supposed to
// suppress the implicit instantiation of non-inline member functions. Make sure
// that we suppress the implicit instantiation of non-inline member functions
// defined out-of-line. That we aren't instantiating the basic_string
// constructor when we shouldn't be. Such an instantiation forces the implicit
// instantiation of _S_construct<const char*>. Since _S_construct is a member
// template, it's instantiation is *not* suppressed (despite being in
// basic_string<char>), so we would emit it as a weak definition.

#define _LIBCPP_EXCEPTION_ABI __attribute__ ((__visibility__("default")))
#define _LIBCPP_INLINE_VISIBILITY __attribute__ ((__visibility__("hidden"), __always_inline__))
#define _LIBCPP_VISIBLE __attribute__ ((__visibility__("default")))
#if (__has_feature(cxx_noexcept))
#  define _NOEXCEPT noexcept
#  define _NOEXCEPT_(x) noexcept(x)
#else
#  define _NOEXCEPT throw()
#  define _NOEXCEPT_(x)
#endif

namespace std  // purposefully not using versioning namespace
{

template<class charT> struct char_traits;
template<class T>     class allocator;
template <class _CharT,
          class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class _LIBCPP_VISIBLE basic_string;
typedef basic_string<char, char_traits<char>, allocator<char> > string;

class _LIBCPP_EXCEPTION_ABI exception
{
public:
    _LIBCPP_INLINE_VISIBILITY exception() _NOEXCEPT {}
    virtual ~exception() _NOEXCEPT;
    virtual const char* what() const _NOEXCEPT;
};

class _LIBCPP_EXCEPTION_ABI runtime_error
    : public exception
{
private:
    void* __imp_;
public:
    explicit runtime_error(const string&);
    explicit runtime_error(const char*);

    runtime_error(const runtime_error&) _NOEXCEPT;
    runtime_error& operator=(const runtime_error&) _NOEXCEPT;

    virtual ~runtime_error() _NOEXCEPT;

    virtual const char* what() const _NOEXCEPT;
};

}

void dummysymbol() {
  throw(std::runtime_error("string"));
}

namespace not_weak_on_first {
  int func();
  // CHECK: {{.*}} extern_weak {{.*}} @_ZN17not_weak_on_first4funcEv(
  int func() __attribute__ ((weak));

  typedef int (*FuncT)();

  extern const FuncT table[] = {
      func,
  };
}

namespace constant_eval {
  [[gnu::weak]] extern int a;
  // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
  // CHECK:     [[CMP:%.*]] = icmp ne ptr @_ZN13constant_eval1aE, null
  // CHECK:     [[ZEXT:%.*]] = zext i1 [[CMP]] to i8
  // CHECK:     store i8 [[ZEXT]], ptr @_ZN13constant_eval6has_a1E,
  bool has_a1 = &a;
  // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
  // CHECK:     [[CMP:%.*]] = icmp ne ptr @_ZN13constant_eval1aE, null
  // CHECK:     [[ZEXT:%.*]] = zext i1 [[CMP]] to i8
  // CHECK:     store i8 [[ZEXT]], ptr @_ZN13constant_eval6has_a2E,
  bool has_a2 = &a != nullptr;

  struct X {
    [[gnu::weak]] void f();
  };
  // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
  // CHECK:     [[CMP:%.*]] = icmp ne i{{32|64}} ptrtoint (ptr @_ZN13constant_eval1X1fEv to i{{32|64}}), 0
  // CHECK:     [[ZEXT:%.*]] = zext i1 [[CMP]] to i8
  // CHECK:     store i8 [[ZEXT]], ptr @_ZN13constant_eval6has_f1E,
  bool has_f1 = &X::f;
  // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
  // CHECK:     [[CMP:%.*]] = icmp ne i{{32|64}} ptrtoint (ptr @_ZN13constant_eval1X1fEv to i{{32|64}}), 0
  // CHECK:     [[CMP2:%.*]] = icmp ne i{{32|64}} ptrtoint (ptr @_ZN13constant_eval1X1fEv to i{{32|64}}), 0
  // CHECK:     [[AND:%.*]] = and i1 [[CMP2]], false
  // CHECK:     [[OR:%.*]] = or i1 [[CMP]], [[AND]]
  // CHECK:     [[ZEXT:%.*]] = zext i1 [[OR]] to i8
  // CHECK:     store i8 [[ZEXT]], ptr @_ZN13constant_eval6has_f2E,
  bool has_f2 = &X::f != nullptr;
}
