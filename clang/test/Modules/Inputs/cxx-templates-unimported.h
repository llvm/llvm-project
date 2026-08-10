#include "cxx-templates-common.h"

namespace hidden_specializations {
  // explicit specializations
  template<> void fn<int>() {}
  template<> struct cls<int> {
    void nested_fn();
    struct nested_cls;
    static int nested_var;
    enum nested_enum : int;
  };
  template<> int var<int>;

  // partial specializations
  template<typename T> struct cls<T*> {
    void nested_fn();
    struct nested_cls;
    static int nested_var;
    enum nested_enum : int;
  };
  template<typename T> int var<T*>;

  // member specializations
  template<> void cls<void>::nested_fn() {}
  template<> struct cls<void>::nested_cls {};
  template<> int cls<void>::nested_var;
  template<> enum class cls<void>::nested_enum { e };
  template<> template<typename U> void cls<void>::nested_fn_t() {}
  template<> template<typename U> struct cls<void>::nested_cls_t {};
  template<> template<typename U> int cls<void>::nested_var_t;

  // specializations instantiated here are ok if their pattern is
  inline void use_stuff() {
    fn<char>();
    cls<char>();
    (void)var<char>;
    cls<char*>();
    (void)var<char*>;
    cls<void>::nested_fn_t<char>();
    cls<void>::nested_cls_t<char>();
    (void)cls<void>::nested_var_t<char>;
  }
}
