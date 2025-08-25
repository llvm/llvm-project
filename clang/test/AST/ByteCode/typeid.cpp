// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20                                         -verify=ref,both %s

namespace std {
struct __type_info_implementations {
  struct __string_impl_base {
    typedef const char *__type_name_t;
  };
  struct __unique_impl : __string_impl_base {

    static bool __eq(__type_name_t __lhs, __type_name_t __rhs);
  };
  typedef __unique_impl __impl;
};

class __pointer_type_info {
public:
  int __flags = 0;
};

class type_info : public __pointer_type_info {
protected:
  typedef __type_info_implementations::__impl __impl;
  __impl::__type_name_t __type_name;
};
}; // namespace std

static_assert(&typeid(int) != &typeid(long));
static_assert(&typeid(int) == &typeid(int));
static_assert(&typeid(int) < &typeid(long)); // both-error {{not an integral constant expression}} \
                                             // both-note {{comparison between pointers to unrelated objects '&typeid(int)' and '&typeid(long)' has unspecified value}}
static_assert(&typeid(int) > &typeid(long)); // both-error {{not an integral constant expression}} \
                                             // both-note {{comparison between pointers to unrelated objects '&typeid(int)' and '&typeid(long)' has unspecified value}}

 struct Base {
   virtual void func() ;
 };
 struct Derived : Base {};

constexpr bool test() {
  Derived derived;
  Base const &as_base = derived;
  if (&typeid(as_base) != &typeid(Derived))
    __builtin_abort();
  return true;
}
static_assert(test());

int dontcrash() {
  auto& pti = static_cast<const std::__pointer_type_info&>(
      typeid(int)
  );
  return pti.__flags == 0 ? 1 : 0;
}
