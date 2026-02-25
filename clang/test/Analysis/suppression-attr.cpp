// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

struct Clazz {
  template <typename T>
  static void templated_memfn();
};

// This must come before the 'templated_memfn' is defined!
static void instantiate() {
  Clazz::templated_memfn<int>();
}

template <typename T>
void Clazz::templated_memfn() {
  // When we report a bug in a function, we traverse the lexical decl context
  // of it while looking for suppression attributes to record what source
  // ranges should the suppression apply to.
  // In the past, that traversal didn't follow template instantiations, only
  // primary templates.
  [[clang::suppress]] clang_analyzer_warnIfReached(); // no-warning

}

namespace [[clang::suppress]]
suppressed_namespace {
  int foo() {
    int *x = 0;
    return *x;
  }

  int foo_forward();
}

int suppressed_namespace::foo_forward() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

// Another instance of the same namespace.
namespace suppressed_namespace {
  int bar() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}

void lambda() {
  [[clang::suppress]] {
    auto lam = []() {
      int *x = 0;
      return *x;
    };
  }
}

class [[clang::suppress]] SuppressedClass {
  int foo() {
    int *x = 0;
    return *x;
  }

  int bar();
};

int SuppressedClass::bar() {
  int *x = 0;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

class SuppressedMethodClass {
  [[clang::suppress]] int foo() {
    int *x = 0;
    return *x;
  }

  [[clang::suppress]] int bar1();
  int bar2();
};

int SuppressedMethodClass::bar1() {
  int *x = 0;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

[[clang::suppress]]
int SuppressedMethodClass::bar2() {
  int *x = 0;
  return *x; // no-warning
}


template <class> struct S1 {
  template <class> struct S2 {
    int i;
    template <class T> int m(const S2<T>& s2) {
      return s2.i; // expected-warning{{Undefined or garbage value returned to caller}}
    }
  };
};

void gh_182659() {
  S1<int>::S2<int> s1;
  S1<int>::S2<char> s2;
  s1.m(s2);
}

template <typename T>
class [[clang::suppress]] ClassTemplateAttrOnClass {
public:
  void inline_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  void out_of_line_method();
};

template <typename T>
void ClassTemplateAttrOnClass<T>::out_of_line_method() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
struct ClassTemplateAttrOnOutOfLineDef {
  void method();
};

template <typename T>
[[clang::suppress]]
void ClassTemplateAttrOnOutOfLineDef<T>::method() {
  clang_analyzer_warnIfReached(); // no-warning
}

template <typename T>
struct ClassTemplateAttrOnDecl {
  [[clang::suppress]] void method();
};

template <typename T>
void ClassTemplateAttrOnDecl<T>::method() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_template_class() {
  ClassTemplateAttrOnClass<int>().inline_method();
  ClassTemplateAttrOnClass<int>().out_of_line_method();
  ClassTemplateAttrOnOutOfLineDef<int>().method();
  ClassTemplateAttrOnDecl<int>().method();
}

// Just the declaration.
template <typename T> [[clang::suppress]] void FunctionTemplateSuppressed(T);
template <typename T>
void FunctionTemplateSuppressed(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
void FunctionTemplateUnsuppressed(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_function_templates() {
  FunctionTemplateSuppressed(0);
  FunctionTemplateUnsuppressed(0);
}

// Only the <int*> specialization carries the attribute.
template <typename T>
struct ExplicitFullClassSpecializationAttrOnSpec {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

template <>
class [[clang::suppress]] ExplicitFullClassSpecializationAttrOnSpec<int *> {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Only the primary template carries the attribute.  The explicit
// specialization is a completely independent class and is NOT suppressed.
template <typename T>
class [[clang::suppress]] ExplicitFullClassSpecializationAttrOnPrimary {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

template <>
struct ExplicitFullClassSpecializationAttrOnPrimary<int *> {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void instantiate_full_spec_class() {
  ExplicitFullClassSpecializationAttrOnSpec<long>().method();   // warns (primary)
  ExplicitFullClassSpecializationAttrOnSpec<int *>().method();  // suppressed (explicit specialization)

  ExplicitFullClassSpecializationAttrOnPrimary<long>().method();   // suppressed (primary)
  ExplicitFullClassSpecializationAttrOnPrimary<int *>().method();  // warns (explicit specialization)
}

// Only the <int *> specialization is suppressed.
template <typename T>
void ExplicitFullFunctionSpecializationAttrOnSpec(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <>
[[clang::suppress]] void ExplicitFullFunctionSpecializationAttrOnSpec(int *) {
  clang_analyzer_warnIfReached(); // no-warning
}

// Only the primary template is suppressed.
template <typename T>
[[clang::suppress]] void ExplicitFullFunctionSpecializationAttrOnPrimary(T) {
  clang_analyzer_warnIfReached(); // no-warning
}

template <>
void ExplicitFullFunctionSpecializationAttrOnPrimary(int *) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_full_spec_function() {
  ExplicitFullFunctionSpecializationAttrOnSpec(0L);      // warns (primary)
  ExplicitFullFunctionSpecializationAttrOnSpec((int *)nullptr); // suppressed (explicit specialization)

  ExplicitFullFunctionSpecializationAttrOnPrimary(0L);      // suppressed (primary)
  ExplicitFullFunctionSpecializationAttrOnPrimary((int *)nullptr); // warns (explicit specialization)
}

// Only the <T, int *> partial specialization carries the attribute.
template <typename T, typename U>
struct PartialClassSpecializationAttrOnPartial {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

template <typename T>
class [[clang::suppress]] PartialClassSpecializationAttrOnPartial<T, int *> {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Only the primary template carries the attribute; partial spec is separate.
template <typename T, typename U>
class [[clang::suppress]] PartialClassSpecializationAttrOnPrimary {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

template <typename T>
struct PartialClassSpecializationAttrOnPrimary<T, int *> {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void instantiate_partial_spec() {
  PartialClassSpecializationAttrOnPartial<long, long>().method();    // warns (primary)
  PartialClassSpecializationAttrOnPartial<long, int *>().method();   // suppressed (partial spec)

  PartialClassSpecializationAttrOnPrimary<long, long>().method();    // suppressed (primary)
  PartialClassSpecializationAttrOnPrimary<long, int *>().method();   // warns (partial spec)
}

// Attribute on outer -> suppresses both outer and inner inline methods.
template <typename T>
class [[clang::suppress]] NestedTemplateClassAttrOnOuter {
public:
  void outer_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }

  template <typename U>
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

// Attribute on inner only -> outer method NOT suppressed.
template <typename T>
struct NestedTemplateClassAttrOnInner {
  void outer_method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }

  template <typename U>
  class [[clang::suppress]] Inner {
  public:
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

void instantiate_nested() {
  NestedTemplateClassAttrOnOuter<int>().outer_method();
  NestedTemplateClassAttrOnOuter<int>::Inner<long>().method();

  NestedTemplateClassAttrOnInner<int>().outer_method();
  NestedTemplateClassAttrOnInner<int>::Inner<long>().method();
}

struct NonTemplateClassWithTemplatedMethod {
  template <typename T>
  [[clang::suppress]] void suppressed(T) {
    clang_analyzer_warnIfReached(); // no-warning
  }

  template <typename T>
  void unsuppressed(T) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void instantiate_nontpl_templated_method() {
  NonTemplateClassWithTemplatedMethod obj;
  obj.suppressed(0);
  obj.unsuppressed(0);
}

template <typename T>
struct TemplateClassWithTemplateMethod {
  template <typename U>
  [[clang::suppress]] void suppressed(U) {
    clang_analyzer_warnIfReached(); // no-warning
  }

  template <typename U>
  void unsuppressed(U) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }

  template <typename U>
  [[clang::suppress]] void suppress_at_decl_outline(U);

  template <typename U>
  void suppress_at_def_outline(U);
};

template <typename T>
template <typename U>
void TemplateClassWithTemplateMethod<T>::suppress_at_decl_outline(U) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
template <typename U>
[[clang::suppress]] void TemplateClassWithTemplateMethod<T>::suppress_at_def_outline(U) {
  clang_analyzer_warnIfReached(); // no-warning
}

void instantiate_tpl_class_tpl_method() {
  TemplateClassWithTemplateMethod<int> obj;
  obj.suppressed(0L);
  obj.unsuppressed(0L);
  obj.suppress_at_decl_outline(0L);
  obj.suppress_at_def_outline(0L);
}

// A simple "box" template used as a template-template argument.
template <typename T>
struct Box {
  void get() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

// A version of Box that suppresses its own methods.
template <typename T>
class [[clang::suppress]] SuppressedBox {
public:
  void get() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Adaptor whose own methods are suppressed; the contained Box's methods are not.
template <typename T, template <typename> class Container>
class [[clang::suppress]] SuppressedAdaptor {
public:
  Container<T> data;

  void adaptor_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Adaptor with no suppression; Box's own suppression is independent.
template <typename T, template <typename> class Container>
struct UnsuppressedAdaptor {
  Container<T> data;

  void adaptor_method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void instantiate_template_template() {
  // SuppressedAdaptor<Box>: adaptor method suppressed; Box::get not affected.
  SuppressedAdaptor<int, Box> sa;
  sa.adaptor_method();  // suppressed by adaptor's attr
  sa.data.get();        // warns — Box has no attr, different lexical context

  // UnsuppressedAdaptor<SuppressedBox>: adaptor warns; SuppressedBox::get suppressed.
  UnsuppressedAdaptor<int, SuppressedBox> ua;
  ua.adaptor_method();  // warns — adaptor has no attr
  ua.data.get();        // suppressed by SuppressedBox's attr
}

template <typename... Args>
[[clang::suppress]] void Variadic_Suppressed(Args...) {
  clang_analyzer_warnIfReached(); // no-warning
}

// Variadic template function specialization.
template <>
void Variadic_Suppressed(int, long) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_variadic() {
  Variadic_Suppressed();
  Variadic_Suppressed(0);
  Variadic_Suppressed(0, 0L);
}

// 3 levels of nesting:
// The suppression mechanism walks the member-template-instantiation chain.
// Verify it reaches the primary template definition at depth 3.
// Similar to gh_182659.

template <typename A>
struct [[clang::suppress]] ThreeLevels_AttrOnOuter {
  template <typename B>
  struct Mid {
    template <typename C>
    struct Inner {
      void inline_defined() {
        clang_analyzer_warnIfReached(); // no-warning
      }
      void outline_defined();
    };
  };
};

template <typename A>
template <typename B>
template <typename C>
void ThreeLevels_AttrOnOuter<A>::Mid<B>::Inner<C>::outline_defined() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename A>
struct ThreeLevels_AttrOnInner {
  template <typename B>
  struct Mid {
    template <typename C>
    struct [[clang::suppress]] Inner {
      void inline_defined() {
        clang_analyzer_warnIfReached(); // no-warning
      }
      void outline_defined();
    };
  };
};

template <typename A>
template <typename B>
template <typename C>
void ThreeLevels_AttrOnInner<A>::Mid<B>::Inner<C>::outline_defined() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename A>
struct ThreeLevels_NoAttr {
  template <typename B>
  struct Mid {
    template <typename C>
    struct Inner {
      void inline_defined() {
        clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      }
      void outline_defined();
    };
  };
};

template <typename A>
template <typename B>
template <typename C>
void ThreeLevels_NoAttr<A>::Mid<B>::Inner<C>::outline_defined() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_three_levels() {
  ThreeLevels_AttrOnOuter<int>::Mid<long>::Inner<short>().inline_defined();
  ThreeLevels_AttrOnOuter<int>::Mid<long>::Inner<short>().outline_defined();

  ThreeLevels_AttrOnInner<int>::Mid<long>::Inner<short>().inline_defined();
  ThreeLevels_AttrOnInner<int>::Mid<long>::Inner<short>().outline_defined();

  ThreeLevels_NoAttr<int>::Mid<long>::Inner<short>().inline_defined();
  ThreeLevels_NoAttr<int>::Mid<long>::Inner<short>().outline_defined();
}

template <typename T>
class [[clang::suppress]] ClassTemplateStaticMethod {
public:
  static void static_method_inline() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  static void static_method_outline();
};

template <typename T>
void ClassTemplateStaticMethod<T>::static_method_outline() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
struct ClassTemplateStaticMethod_NoAttr {
  static void static_method_inline() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
  static void static_method_outline();
};

template <typename T>
void ClassTemplateStaticMethod_NoAttr<T>::static_method_outline() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_static_methods() {
  ClassTemplateStaticMethod<int>::static_method_inline();
  ClassTemplateStaticMethod<int>::static_method_outline();
  ClassTemplateStaticMethod_NoAttr<int>::static_method_inline();
  ClassTemplateStaticMethod_NoAttr<int>::static_method_outline();
}

// Forward declarations for the friend declarations so that they can be called without ADL.
extern void friend_inline_in_suppressed_class();
extern void friend_ool_in_suppressed_class();
struct [[clang::suppress]] Friend_SuppressedClass {
  friend void friend_inline_in_suppressed_class() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  friend void friend_ool_in_suppressed_class();
};

void friend_ool_in_suppressed_class() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

extern void friend_inline_in_unsuppressed_class();
struct Friend_UnsuppressedClass {
  friend void friend_inline_in_unsuppressed_class() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

// Out-of-line definition with the attribute placed on the definition itself.
extern void friend_attr_on_ool_def();
struct Friend_AttrOnDef {
  friend void friend_attr_on_ool_def();
};
[[clang::suppress]]
void friend_attr_on_ool_def() {
  clang_analyzer_warnIfReached(); // no-warning
}

// Friend function template defined inline in a suppressed class.
template <typename T>
extern void friend_template_in_suppressed_class(T);
template <typename T>
extern void friend_template_ool_in_suppressed_class(T);
struct [[clang::suppress]] Friend_SuppressedClassWithTemplate {
  template <typename T>
  friend void friend_template_in_suppressed_class(T) {
    // FIXME: This should be suppressed.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }

  template <typename T>
  friend void friend_template_ool_in_suppressed_class(T);
};

template <typename T>
extern void friend_template_ool_in_suppressed_class(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
extern void friend_template_in_unsuppressed_class(T);
template <typename T>
extern void friend_template_ool_in_unsuppressed_class(T);
struct Friend_UnsuppressedClassWithTemplate {
  template <typename T>
  friend void friend_template_in_unsuppressed_class(T) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
  template <typename T>
  friend void friend_template_ool_in_unsuppressed_class(T);
};

template <typename T>
void friend_template_ool_in_unsuppressed_class(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void instantiate_friends() {
  friend_inline_in_suppressed_class();
  friend_ool_in_suppressed_class();

  friend_inline_in_unsuppressed_class();
  friend_attr_on_ool_def();

  friend_template_in_suppressed_class(0);
  friend_template_ool_in_suppressed_class(0);

  friend_template_in_unsuppressed_class(0);
  friend_template_ool_in_unsuppressed_class(0);
}
