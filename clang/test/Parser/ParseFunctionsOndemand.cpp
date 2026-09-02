// RUN: %clang_cc1 -fparse-functions-ondemand -fsyntax-only -verify -Wall -Wextra -Werror -ferror-limit 0 -std=c++11 %s
// RUN: not %clang_cc1 -fparse-functions-ondemand -fdelayed-template-parsing -fsyntax-only -std=c++11 %s 2>&1 | FileCheck %s --check-prefix=MUTEX

// MUTEX: error: invalid argument '-fparse-functions-ondemand' not allowed with '-fdelayed-template-parsing'

// Internal-linkage bodies are parsed only when referenced; external ones eagerly.
namespace InternalFunctionBodies {
static void static_diagnoses_on_use() {
  undeclared_static(); // expected-error {{use of undeclared identifier 'undeclared_static'}}
}

static void prior_static();
void prior_static() {
  undeclared_prior_static(); // expected-error {{use of undeclared identifier 'undeclared_prior_static'}}
}

static void unreferenced_prior_static();
void unreferenced_prior_static() {
  undeclared_prior_static(); // this should not give an error
}

namespace {
void anon_namespace_function() {
  undeclared_anon_namespace(); // expected-error {{use of undeclared identifier 'undeclared_anon_namespace'}}
}

void unreferenced_anon_namespace_function() {
  undeclared_anon_namespace(); // this should not give an error
}
} // namespace

void external_function_diagnoses_eagerly() {
  undeclared_external(); // expected-error {{use of undeclared identifier 'undeclared_external'}}
}

static void unreferenced_static_body_is_delayed() {
  undeclared_but_unreferenced(); // this should not give an error
}

__attribute__((used)) static int attribute_used_body_is_parsed() {
  return undeclared_attribute_used(); // expected-error {{use of undeclared identifier 'undeclared_attribute_used'}}
}

void trigger_internal_function_bodies() {
  static_diagnoses_on_use();
  prior_static();
  anon_namespace_function();
}
} // namespace InternalFunctionBodies

// A delayed body resolves overloads using only those declared before it.
namespace SourceOrderOverloads {
struct One {};
struct Two {}; // expected-note 2 {{candidate constructor}}

static One foo_hidden_later(int);
static void call_before_later_overload() {
  Two value = foo_hidden_later(0.0); // expected-error {{no viable conversion from 'One' to 'Two'}}
}
static Two foo_hidden_later(double);

static Two foo_forward_declared(double);
static void call_forward_declared_overload() {
  Two _ = foo_forward_declared(0.0);
}
static Two foo_forward_declared(double);

static One foo_both_before(int);
static Two foo_both_before(double);
static void call_both_overloads_before() {
  Two _ = foo_both_before(0.0);
}

void trigger_source_order_overloads() {
  call_before_later_overload();
  call_forward_declared_overload();
  call_both_overloads_before();
}
} // namespace SourceOrderOverloads

// A delayed body sees namespace-scope names only if declared before it.
namespace NamespaceScopeSourceOrder {
static int later_variable_hidden() {
  return later_variable; // expected-error {{use of undeclared identifier 'later_variable'}}
}
static int later_variable;

static int later_function_hidden() {
  return later_function(); // expected-error {{use of undeclared identifier 'later_function'}}
}
static int later_function();

static int builtin_lookup_still_works() {
  return __builtin_abs(-1);
}

void trigger_namespace_scope_source_order() {
  later_variable_hidden();
  later_function_hidden();
  builtin_lookup_still_works();
}
} // namespace NamespaceScopeSourceOrder

// Member bodies are delayed too; when parsed they see the whole class but only
// prior namespace-scope names.
namespace ClassMethodLookup {
struct NormalClass {
  void sees_later_method() {
    later_method();
  }
  void later_method();

  int sees_later_static_member() {
    return later_static_member;
  }
  static int later_static_member;
};

namespace {
struct InternalClass {
  void unreferenced_body_is_delayed() {
    undeclared_in_method(); // this should not give an error
  }

  void sees_later_member() {
    later_member();
  }
  void later_member() {}

  void diagnoses_on_use() {
    undeclared_in_referenced_method(); // expected-error {{use of undeclared identifier 'undeclared_in_referenced_method'}}
  }

  void later_global_hidden() {
    later_global_after_class; // expected-error {{use of undeclared identifier 'later_global_after_class'}}
  }

  static int static_member_later_global_hidden() {
    return later_global_after_class; // expected-error {{use of undeclared identifier 'later_global_after_class'}}
  }

  static int static_member_sees_later_static_member() {
    return later_static_member;
  }
  static int later_static_member;
};
} // namespace

int InternalClass::later_static_member = 10;
int later_global_after_class = 10;

void trigger_class_method_lookup() {
  InternalClass object;
  object.sees_later_member();
  object.diagnoses_on_use();
  object.later_global_hidden();
  InternalClass::static_member_later_global_hidden();
  InternalClass::static_member_sees_later_static_member();
}
} // namespace ClassMethodLookup

// Out-of-line member bodies see namespace-scope names declared before the
// out-of-line definition.
namespace OutOfClassMethods {
namespace {
struct S {
  int sees_prior_global();
  int hides_later_global();
};

int prior_global; // expected-note {{'prior_global' declared here}}

int S::sees_prior_global() {
  return prior_global;
}

int S::hides_later_global() {
  return later_global; // expected-error {{use of undeclared identifier 'later_global'}}
}

int later_global; // expected-note 2 {{'OutOfClassMethods::later_global' declared here}}
} // namespace

void trigger_out_of_class_methods() {
  S s;
  s.sees_prior_global();
  s.hides_later_global();
}
} // namespace OutOfClassMethods

// Inline friend and friend-class member bodies follow the same prior-only
// namespace-scope visibility.
namespace FriendFunctionsAndClasses {
int prior_global;

namespace {
struct FriendFunctionPrior {
  friend int friend_function_prior(FriendFunctionPrior) {
    return prior_global;
  }
};

struct FriendFunctionLater {
  friend int friend_function_later(FriendFunctionLater) {
    return later_global; // expected-error {{use of undeclared identifier 'later_global'}}
  }
};

struct Host {
  friend struct FriendClass;
};

struct FriendClass {
  static int sees_prior_global() {
    return prior_global;
  }
  static int hides_later_global() {
    return later_global; // expected-error {{use of undeclared identifier 'later_global'}}
  }
};
} // namespace

int later_global;

void trigger_friend_functions_and_classes() {
  friend_function_prior(FriendFunctionPrior{});
  friend_function_later(FriendFunctionLater{});
  FriendClass::sees_prior_global();
  FriendClass::hides_later_global();
}
} // namespace FriendFunctionsAndClasses

// ADL from a delayed body finds only functions declared before it.
namespace ADLLookup {
struct One {};
struct Two {};

namespace Hidden {
struct A {};
}

static void adl_later_function_hidden() {
  Hidden::A a;
  Two value = adl_target(a); // expected-error {{use of undeclared identifier 'adl_target'}}
}

namespace Hidden {
One adl_target(A);
}

namespace Visible {
struct A {};
One adl_target(A);
}

static void adl_prior_function_visible() {
  Visible::A a;
  One _ = adl_target(a);
}

void trigger_adl_lookup() {
  adl_later_function_hidden();
  adl_prior_function_visible();
}
} // namespace ADLLookup

// Qualified lookup from a delayed body finds only members declared before it.
namespace QualifiedLookup {
struct One {};
struct Two {};

namespace Hidden {}

static void qualified_later_function_hidden() {
  Hidden::h(); // expected-error {{no member named 'h' in namespace 'QualifiedLookup::Hidden'}}
}

namespace Hidden {
One h();
}

namespace Visible {
One h();
}

static void qualified_prior_function_visible() {
  One _ = Visible::h();
}

void trigger_qualified_lookup() {
  qualified_later_function_hidden();
  qualified_prior_function_visible();
}
} // namespace QualifiedLookup

// Default arguments, noexcept, and trailing-return are parsed eagerly (at the
// declaration), so they cannot see later names; delete/default bodies are fine.
namespace DeclarationParts {
static int default_argument(int value = default_argument_later_global) { // expected-error {{use of undeclared identifier 'default_argument_later_global'}}
  return value;
}
int default_argument_later_global;

static void noexcept_expr() noexcept(noexcept(noexcept_later_global)) {} // expected-error {{use of undeclared identifier 'noexcept_later_global'}}
int noexcept_later_global;

static auto trailing_return() -> decltype(trailing_return_later_global) { // expected-error {{use of undeclared identifier 'trailing_return_later_global'}}
  return 0;
}
int trailing_return_later_global;

void deleted_function() = delete;

struct DefaultedConstructor {
  DefaultedConstructor() = default;
};
} // namespace DeclarationParts

// Local-class member bodies are parsed with their enclosing function's body.
namespace LocalClassCases {
void local_class_method_diagnoses_on_use() {
  struct S {
    void f() {
      undeclared_in_local_class(); // expected-error {{use of undeclared identifier 'undeclared_in_local_class'}}
    }
  };

  S s;
  s.f();
}

void local_class_method_sees_local_typedef() {
  typedef int T;
  struct S {
    T f() { return 0; }
  };

  S s;
  s.f();
}

// A local-class member body can name a sibling local declaration, so it must be
// parsed with its enclosing function rather than deferred and re-parsed at the
// end of the TU (where the enclosing scope is gone).
void local_class_method_sees_sibling_local_class() {
  struct Helper {
    int value() { return 7; }
  };
  struct User {
    int use() {
      Helper h; // must still resolve 'Helper'
      return h.value();
    }
  };

  User u;
  u.use();
}

// Same, but the referencing member is reached through the vtable of a locally
// constructed object.
void local_class_virtual_method_sees_sibling_local_class() {
  struct Helper {
    int value() { return 7; }
  };
  struct Base {
    virtual int get() { return 0; }
    virtual ~Base() {}
  };
  struct Derived : Base {
    int get() override {
      Helper h; // must still resolve 'Helper'
      return h.value();
    }
  };

  Derived d;
  (void)d;
}
} // namespace LocalClassCases

// A hidden friend defined inline in a class template is late-parsed on demand.
// Its body is parsed in the complete-class context, so it must see members of
// the enclosing class regardless of source order -- including members declared
// after it, as in libstdc++'s forward_list iterator comparison operators.
namespace HiddenFriendMemberOrder {
template <class T>
struct Iter {
  friend bool operator==(const Iter &a, const Iter &b) {
    return a.node == b.node; // 'node' is declared later in the same class
  }
  friend bool operator!=(const Iter &a, const Iter &b) {
    return a.node != b.node;
  }
  void *node;
};

bool trigger_hidden_friend_member_order() {
  Iter<int> a{}, b{};
  return a == b || a != b;
}
} // namespace HiddenFriendMemberOrder

namespace VirtualFunctions {
// Constructing an object of a polymorphic internal-linkage class emits its
// vtable, which references every virtual function. So all virtual function
// bodies are parsed when the object is used, while unreferenced non-virtual
// members stay unparsed.
namespace {
struct UsedObject {
  virtual void first_virtual() {
    undeclared_in_first_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_first_virtual'}}
  }
  virtual void second_virtual() {
    undeclared_in_second_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_second_virtual'}}
  }
  void unreferenced_non_virtual() {
    undeclared_in_non_virtual(); // this should not give an error
  }
};

// No object is ever created, so none of the virtual bodies are parsed.
struct NeverInstantiated {
  virtual void unused_first_virtual() {
    undeclared_in_unused_first(); // this should not give an error
  }
  virtual void unused_second_virtual() {
    undeclared_in_unused_second(); // this should not give an error
  }
};

// Only a pointer is formed; without constructing an object the vtable is not
// emitted and the virtual bodies stay unparsed.
struct OnlyPointerUsed {
  virtual void pointer_virtual() {
    undeclared_in_pointer_virtual(); // this should not give an error
  }
};

// Calling a single virtual function still emits the vtable, so every virtual
// body is parsed -- not just the one that was called.
struct CallOneVirtual {
  virtual void called_virtual() {
    undeclared_in_called_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_called_virtual'}}
  }
  virtual void other_virtual() {
    undeclared_in_other_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_other_virtual'}}
  }
};

// Using a derived object emits both the derived and base vtables, so virtual
// bodies from the whole hierarchy are parsed.
struct Base {
  virtual void base_virtual() {
    undeclared_in_base_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_base_virtual'}}
  }
};

struct Derived : Base {
  void base_virtual() override {
    undeclared_in_override_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_override_virtual'}}
  }
  virtual void derived_virtual() {
    undeclared_in_derived_virtual(); // expected-error {{use of undeclared identifier 'undeclared_in_derived_virtual'}}
  }
};
} // namespace

void trigger_virtual_functions() {
  UsedObject used;
  (void)used;

  CallOneVirtual call;
  call.called_virtual();

  Derived derived;
  (void)derived;

  OnlyPointerUsed *pointer = nullptr;
  (void)pointer;
}
} // namespace VirtualFunctions

// A vtable whose first use is inside an on-demand-deferred internal function is
// only marked used while that body is parsed at the end of the TU, after the
// initial vtable-defining pass. Its virtual bodies must still be parsed (here,
// diagnosed) rather than emitted unparsed.
namespace VTableUsedInDeferredFunction {
namespace {
struct Polymorphic {
  virtual int poly_method() {
    undeclared_in_poly_method(); // expected-error {{use of undeclared identifier 'undeclared_in_poly_method'}}
    return 0;
  }
};
}

// Internal linkage, so this body is deferred; constructing the object here is
// the first use of Polymorphic's vtable.
static Polymorphic &get_instance() {
  static Polymorphic instance;
  return instance;
}

void trigger_vtable_used_in_deferred_function() {
  get_instance();
}
} // namespace VTableUsedInDeferredFunction

// An explicit function-template specialization is not TK_NonTemplate, so the
// on-demand machinery cannot re-parse it. Even when it has internal linkage
// (its template argument is an anonymous-namespace type) it must be parsed
// eagerly rather than deferred; otherwise its body (here, diagnosed) would be
// left unparsed and emitted empty.
namespace ExplicitSpecialization {
namespace {
struct InternalField {};
struct InternalField2 {};
}
struct Parser {
  template <class T> bool parse(T &field);
};
template <>
bool Parser::parse(InternalField &field) {
  (void)field;
  undeclared_in_explicit_specialization(); // this should not error
  return false;
}
template <>
bool Parser::parse(InternalField2 &field) {
  (void)field;
  undeclared_in_explicit_specialization(); // expected-error {{use of undeclared identifier 'undeclared_in_explicit_specialization'}}
  return false;
}
} // namespace ExplicitSpecialization

void trigger_explici_specialization() {
    ExplicitSpecialization::Parser p;
    ExplicitSpecialization::InternalField2 f;
    p.parse(f);
}

// A function template's body is deferred on demand (it is late-parsed), but its
// signature is still substituted at call sites. That substitution -- here SFINAE
// on decltype(Builder()) -- must not be subject to the late-parsed body's
// source-order visibility restriction, otherwise a capturing lambda's call
// operator (declared at the call site) is wrongly hidden and the call fails to
// resolve. (A non-capturing lambda avoids the restriction, so a capturing one is
// used here.)
namespace SFINAEWithCapturingLambda {
struct Diag {
  int x;
};
struct Emitter {
  void emit(Diag &&) {}
  template <class T> void emit(T Builder, decltype(Builder()) * = nullptr) {
    emit(Builder());
  }
};
void trigger_sfinae_with_capturing_lambda(Emitter &E) {
  int local = 0;
  E.emit([&]() { return Diag{local}; });
}
} // namespace SFINAEWithCapturingLambda

// Anonymous-namespace function templates are still parsed on instantiation, and
// dependent lookups in their signatures and bodies must resolve normally.
namespace AnonymousNamespaceTemplateLookup {
namespace {
template <typename T> void call_foo(T &t) { t.foo(); }

template <typename T> typename T::result call_bar(T &t) {
  return t.bar();
}

struct T {
  using result = int;

  void foo() {}

  result bar() { return 0; }
};
} // namespace

void trigger_anonymous_namespace_template_lookup() {
  T t;
  call_foo(t);
  (void)call_bar(t);
}
} // namespace AnonymousNamespaceTemplateLookup

// Template bodies are still parsed on instantiation.
namespace TemplateCases {
template <class T>
void template_body_diagnoses_on_instantiation() {
  undeclared_in_template(); // expected-error {{use of undeclared identifier 'undeclared_in_template'}}
}

void trigger_template_cases() {
  template_body_diagnoses_on_instantiation<int>();
}
} // namespace TemplateCases
