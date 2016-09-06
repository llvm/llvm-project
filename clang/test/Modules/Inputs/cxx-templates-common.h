template<typename T> struct SomeTemplate {};

struct DefinedInCommon {
  void f();
  struct Inner {};
  friend void FoundByADL(DefinedInCommon);
};

template<typename T> struct CommonTemplate {
  enum E { a = 1, b = 2, c = 3 };
};

namespace Std {
  template<typename T> struct WithFriend {
    friend bool operator!=(const WithFriend &A, const WithFriend &B) { return false; }
  };
}

namespace Std {
  template<typename T> void f() {
    extern T g();
  }
}

template<typename T> struct TemplateInstantiationVisibility { typedef int type; };

template<typename T> struct Outer {
  template<typename U> struct Inner {
    static constexpr int f();
    static constexpr int g();
  };
};

template<typename T> struct WithPartialSpecialization {};
template<typename T> struct WithPartialSpecialization<void(T)> { typedef int type; };
typedef WithPartialSpecialization<int*> WithPartialSpecializationUse;
typedef WithPartialSpecialization<void(int)> WithPartialSpecializationUse2;

template<typename T> struct WithExplicitSpecialization;
typedef WithExplicitSpecialization<int> WithExplicitSpecializationUse;

template<typename T> struct WithImplicitSpecialMembers { int n; };

template<typename T> struct WithAliasTemplate {
  template<typename> using X = T;
};

template<typename T> struct WithAnonymousDecls {
  struct { bool k; };
  union { int a, b; };
  struct { int c, d; } s;
  enum { e = 123 };
  typedef int X;
};

namespace hidden_specializations {
  template<typename T> void fn() {}

  template<typename T> struct cls {
    static void nested_fn() {}
    struct nested_cls {};
    static int nested_var;
    enum class nested_enum {};

    template<typename U> static void nested_fn_t() {}
    template<typename U> struct nested_cls_t {};
    template<typename U> static int nested_var_t;
  };

  template<typename T> int var;
}

#include "cxx-templates-textual.h"
