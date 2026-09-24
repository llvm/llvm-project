// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=expected,nomsvc %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=expected,msvc -fms-compatibility -DMSVC_COMPAT %s

typedef enum NamedEnum { NamedValue } NamedAlias; // msvc-note 4 {{declared here}}
typedef enum { AnonymousValue } AnonymousAlias;
typedef enum SameNameEnum { SameNameValue } SameNameEnum;

struct StandardCase {
  enum SameNameEnum *member;
};

namespace StandardUsing {
typedef enum SameName { Value } SameName;
}
using StandardUsing::SameName;

struct StandardUsingCase {
  enum SameName *member;
};

typedef enum FriendNamed { FriendValue } FriendAlias; // #FriendAliasDecl
typedef enum AfterFriendNamed { AfterFriendValue } AfterFriendAlias; // nomsvc-note {{declared here}}
typedef enum TemplateFriendNamed { TemplateFriendValue } TemplateFriendAlias; // expected-note {{declared here}}
typedef enum AfterTemplateFriendNamed { AfterTemplateFriendValue } AfterTemplateFriendAlias; // nomsvc-note {{declared here}}
typedef enum FriendReturnNamed { FriendReturnValue } FriendReturnAlias; // expected-note {{declared here}}
typedef enum TemplateFriendReturnNamed { TemplateFriendReturnValue } TemplateFriendReturnAlias; // expected-note {{declared here}}
typedef enum LateFriendNamed { LateFriendValue } LateFriendAlias; // nomsvc-note {{declared here}}
typedef enum NestedFriendNamed { NestedFriendValue } NestedFriendAlias; // expected-note {{declared here}}
typedef enum NestedInnerNamed { NestedInnerValue } NestedInnerAlias; // nomsvc-note {{declared here}}
typedef enum NestedOuterNamed { NestedOuterValue } NestedOuterAlias; // nomsvc-note {{declared here}}

struct FriendUse {
  friend void friend_alias(enum FriendAlias *);
  // expected-error@-1 {{typedef 'FriendAlias' cannot be referenced with the 'enum' specifier}}
  // expected-note@#FriendAliasDecl {{declared here}}
  enum AfterFriendAlias *member; // nomsvc-error {{typedef 'AfterFriendAlias' cannot be referenced with the 'enum' specifier}}
                                 // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct TemplateFriendUse {
  template <class T>
  friend void template_friend(enum TemplateFriendAlias *); // expected-error {{typedef 'TemplateFriendAlias' cannot be referenced with the 'enum' specifier}}
  enum AfterTemplateFriendAlias *member; // nomsvc-error {{typedef 'AfterTemplateFriendAlias' cannot be referenced with the 'enum' specifier}}
                                         // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct FriendReturnUse {
  friend enum FriendReturnAlias *friend_return(); // expected-error {{typedef 'FriendReturnAlias' cannot be referenced with the 'enum' specifier}}
};

struct TemplateFriendReturnUse {
  template <class T>
  friend enum TemplateFriendReturnAlias *template_friend_return(); // expected-error {{typedef 'TemplateFriendReturnAlias' cannot be referenced with the 'enum' specifier}}
};

struct LateFriendUse {
  enum LateFriendAlias friend *late_friend(); // nomsvc-error {{typedef 'LateFriendAlias' cannot be referenced with the 'enum' specifier}}
                                              // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct NestedFriendUse {
  struct Inner {
    friend void nested_friend(enum NestedFriendAlias *); // expected-error {{typedef 'NestedFriendAlias' cannot be referenced with the 'enum' specifier}}
    enum NestedInnerAlias *member; // nomsvc-error {{typedef 'NestedInnerAlias' cannot be referenced with the 'enum' specifier}}
                                   // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
  };
  enum NestedOuterAlias *member; // nomsvc-error {{typedef 'NestedOuterAlias' cannot be referenced with the 'enum' specifier}}
                                 // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

#ifdef MSVC_COMPAT
struct FunctionParameterUse {
  void unqualified(enum NamedAlias *); // msvc-error {{typedef 'NamedAlias' cannot be referenced with the 'enum' specifier}}
  void qualified(enum ::NamedAlias *); // msvc-error {{typedef 'NamedAlias' cannot be referenced with the 'enum' specifier}}
  void nested(void (*)(enum ::NamedAlias *)); // msvc-error {{typedef 'NamedAlias' cannot be referenced with the 'enum' specifier}}
};

struct Holder {
  enum AnonymousAlias *member; // msvc-warning {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

static_assert(__is_same(decltype(Holder::member), AnonymousAlias *), "");

struct OwnedAlias {
  typedef enum Named { Value } Alias; // nomsvc-note {{declared here}}
  enum Alias *member; // nomsvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
                      // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

static_assert(__is_same(decltype(OwnedAlias::member), OwnedAlias::Alias *), "");

struct AliasBase {
  typedef enum Named { Value } Alias; // nomsvc-note {{declared here}}
};

struct InheritedAlias : AliasBase {
  enum Alias *member; // nomsvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
                      // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

static_assert(__is_same(decltype(InheritedAlias::member), AliasBase::Alias *),
              "");

typedef NamedEnum ChainedAlias; // nomsvc-note {{declared here}}

struct ChainedUse {
  enum ChainedAlias *member; // nomsvc-error {{typedef 'ChainedAlias' cannot be referenced with the 'enum' specifier}}
                             // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

enum class ScopedNamed { ScopedValue };
typedef ScopedNamed ScopedAlias; // nomsvc-note {{declared here}}

struct ScopedUse {
  enum ScopedAlias *member; // nomsvc-error {{typedef 'ScopedAlias' cannot be referenced with the 'enum' specifier}}
                            // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct BodyUse {
  void f() {
    enum NamedAlias *value; // expected-error {{typedef 'NamedAlias' cannot be referenced with the 'enum' specifier}}
    (void)value;
  }
};
#endif

typedef const NamedEnum ConstAlias; // nomsvc-note {{declared here}}

void qualified() {
  enum ::ConstAlias value = NamedValue; // nomsvc-error {{typedef 'ConstAlias' cannot be referenced with the 'enum' specifier}}
                                        // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
                                        // msvc-note@-2 {{variable 'value' declared const here}}
#ifdef MSVC_COMPAT
  static_assert(__is_same(decltype(value), ConstAlias), "");
#endif
  value = NamedValue; // msvc-error {{cannot assign to variable 'value' with const-qualified type 'enum ::ConstAlias'}}
}

using UsingAlias = NamedEnum; // nomsvc-note {{declared here}}

void type_alias() {
  enum ::UsingAlias value; // nomsvc-error {{type alias 'UsingAlias' cannot be referenced with the 'enum' specifier}}
                           // msvc-warning@-1 {{using an 'enum' specifier with a type alias name is a Microsoft extension}}
  (void)value;
}

namespace Qualified {
typedef enum Named { Value } Alias; // nomsvc-note 2 {{declared here}}
}

enum Qualified::Alias *qualified_global; // nomsvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
                                         // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}

namespace QualifiedImport {
using Qualified::Alias;
}

enum QualifiedImport::Alias *qualified_import; // nomsvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
                                               // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}

#ifdef MSVC_COMPAT
static_assert(__is_same(decltype(qualified_import), Qualified::Alias *), "");
#endif

namespace Imported {
typedef enum Named { Value } Alias; // expected-note 2 {{declared here}}
}
using Imported::Alias;

void imported_alias() {
  enum Alias *value; // expected-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
  (void)value;
}

enum Alias *file_using_alias; // expected-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}

struct ClassUsingBase {
  typedef enum Underlying { Value } Alias; // nomsvc-note {{declared here}}
};

struct ClassUsingDerived : ClassUsingBase {
  using ClassUsingBase::Alias;
  enum Alias *member; // nomsvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
                      // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

namespace Directive {
typedef enum Named { Value } DirectiveAlias; // expected-note {{declared here}}
}
using namespace Directive;

struct DirectiveUse {
  enum DirectiveAlias *value; // expected-error {{typedef 'DirectiveAlias' cannot be referenced with the 'enum' specifier}}
};

namespace FileScope {
typedef enum Named { Value } Alias; // expected-note {{declared here}}
enum Alias *value; // expected-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
}

typedef enum LocalNamed { LocalValue } LocalAlias; // expected-note 2 {{declared here}}

void free_unqualified(enum LocalAlias *); // expected-error {{typedef 'LocalAlias' cannot be referenced with the 'enum' specifier}}

void direct_local_alias() {
  enum LocalAlias *value; // expected-error {{typedef 'LocalAlias' cannot be referenced with the 'enum' specifier}}
  (void)value;
}

typedef int IntAlias; // expected-note {{declared here}}

void not_an_enum() {
  enum ::IntAlias value; // expected-error {{typedef 'IntAlias' cannot be referenced with the 'enum' specifier}}
  (void)value;
}

#ifdef MSVC_COMPAT
typedef enum DeprecatedEnum { DeprecatedValue } DeprecatedAlias
    __attribute__((deprecated("old enum alias"))); // msvc-note {{'DeprecatedAlias' has been explicitly marked deprecated here}}

struct DeprecatedUse {
  enum DeprecatedAlias *value; // msvc-warning {{'DeprecatedAlias' is deprecated: old enum alias}}
                               // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct DeprecatedUsingBase {
  typedef enum Underlying { Value } Alias
      __attribute__((deprecated("old class enum alias"))); // msvc-note {{'Alias' has been explicitly marked deprecated here}}
};

struct DeprecatedUsingDerived : DeprecatedUsingBase {
  using DeprecatedUsingBase::Alias;
  enum Alias *value; // msvc-warning {{'Alias' is deprecated: old class enum alias}}
                     // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

namespace Unavailable {
typedef enum Enum { Value } Alias // msvc-note {{'Alias' has been explicitly marked unavailable here}}
    __attribute__((unavailable("unavailable enum alias")));
}

struct UnavailableUse {
  enum Unavailable::Alias *value; // msvc-error {{'Alias' is unavailable: unavailable enum alias}}
                                 // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct PrivateOwner {
private:
  typedef enum Named { Value } Alias; // msvc-note 2 {{declared private here}}
};

enum PrivateOwner::Alias *private_qualified; // msvc-error {{'Alias' is a private member of 'PrivateOwner'}}
                                             // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}

struct PrivateDerived : PrivateOwner {
  enum Alias *private_inherited; // msvc-error {{'Alias' is a private member of 'PrivateOwner'}}
                                 // msvc-warning@-1 {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct PrivateUsingBase {
private:
  typedef enum Underlying { Value } Alias; // msvc-note {{declared private here}}
};

struct PrivateUsingDerived : PrivateUsingBase {
  using PrivateUsingBase::Alias; // msvc-error {{'Alias' is a private member of 'PrivateUsingBase'}}
  enum Alias *private_using; // msvc-warning {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

template <class T>
struct DependentUse {
  enum T::Alias *value; // msvc-warning {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};

struct DependentHost {
  typedef enum Named { Value } Alias; // msvc-note 2 {{declared here}}
};

template struct DependentUse<DependentHost>; // msvc-note {{in instantiation of template class 'DependentUse<DependentHost>' requested here}}

template <class T>
struct DependentParameterUse {
  void f(enum T::Alias *); // msvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
};

template struct DependentParameterUse<DependentHost>; // msvc-note {{in instantiation of template class 'DependentParameterUse<DependentHost>' requested here}}

#ifdef MSVC_COMPAT
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmicrosoft-enum-typedef"
template <class T, class U = enum T::Alias>
struct DependentDefaultArgCarrier {
  U value;
};

template <class T>
void dependent_default_arg_param(DependentDefaultArgCarrier<T>) {}

template void dependent_default_arg_param<DependentHost>(
    DependentDefaultArgCarrier<DependentHost>);
#pragma clang diagnostic pop

struct DependentPackHost {
  typedef enum Named { Value } Alias; // msvc-note {{declared here}}
};

template <class... Ts>
struct DependentPackUse {
  void f(enum Ts::Alias *...); // msvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}
};

template struct DependentPackUse<DependentPackHost>; // msvc-note {{in instantiation of template class 'DependentPackUse<DependentPackHost>' requested here}}
#endif

template <class T>
using DependentFunctionType = void(enum T::Alias *); // msvc-error {{typedef 'Alias' cannot be referenced with the 'enum' specifier}}

using InstantiatedFunctionType = DependentFunctionType<DependentHost>; // msvc-note {{in instantiation of template type alias 'DependentFunctionType' requested here}}

struct DependentSameNameHost {
  typedef enum SameName { Value } SameName;
};

template <class T>
struct DependentSameNameUse {
  enum T::SameName *value;
};

template struct DependentSameNameUse<DependentSameNameHost>;
#endif
