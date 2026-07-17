// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

struct WithCtor { WithCtor(); };
struct Inner { int y; };
struct InnerMarked { int y [[uninit]]; };
struct InnerMixed { int a [[uninit]]; int b; };

struct MissingMember {
  int x; // expected-note {{member 'x' declared here}}
  MissingMember() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};

struct MemInit {
  int x;
  MemInit() : x(0) {}
};

struct DefaultMemberInit {
  int x = 0;
  DefaultMemberInit() {}
};

struct Marked {
  int x [[uninit]];
  Marked() {}
};

struct BodyAssignment {
  int x; // expected-note {{member 'x' declared here}}
  // A plain body assignment is not initialization for this rule.
  BodyAssignment() { x = 0; } // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};

struct NestedAggregate {
  Inner m; // expected-note {{member 'm' declared here}}
  NestedAggregate() {} // expected-error {{constructor does not initialize member 'm' under profile 'std::init'}}
};

// A member whose type's only indeterminate scalar is [[uninit]] is
// acknowledged (paper §6.2), so the constructor need not initialize it.
struct NestedMarkedMember {
  InnerMarked m;
  NestedMarkedMember() {}
};

// A member whose type still leaves an unacknowledged scalar indeterminate fires.
struct NestedMixedMember {
  InnerMixed m; // expected-note {{member 'm' declared here}}
  NestedMixedMember() {} // expected-error {{constructor does not initialize member 'm' under profile 'std::init'}}
};

struct TrustedMemberCtor {
  WithCtor m;
  TrustedMemberCtor() {}
};

struct PartialInit {
  int x;
  int y; // expected-note {{member 'y' declared here}}
  PartialInit() : x(0) {} // expected-error {{constructor does not initialize member 'y' under profile 'std::init'}}
};

struct OutOfLine {
  int x; // expected-note {{member 'x' declared here}}
  OutOfLine();
};
OutOfLine::OutOfLine() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}

template <typename T>
struct Tmpl {
  T x; // expected-note {{member 'x' declared here}}
  Tmpl() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};
template struct Tmpl<int>; // expected-note {{in instantiation of member function 'Tmpl<int>::Tmpl' requested here}}

// A delegating constructor relies on its target, which does initialize the
// member, so nothing fires.
struct Delegating {
  int x;
  Delegating() : Delegating(0) {}
  Delegating(int v) : x(v) {}
};

struct SuppressedCtor {
  int x;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] SuppressedCtor() {}
};

struct SuppressedByRule {
  int x;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ctor_uninit_member")]] SuppressedByRule() {}
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] SuppressedClass {
  int x;
  SuppressedClass() {}
};

// Constructor finalization ignores the parse-time suppress stack (it can fire
// nested in an unrelated instantiation), so a [[profiles::suppress]] on a class
// template must still reach the instantiated constructor through the decl-aware
// walk on the lexical parent -- not the stack pushed while instantiating it.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] SuppressedTemplate {
  T x;
  SuppressedTemplate() {}
};
template struct SuppressedTemplate<int>;

struct Base { int b; };

struct UninitBase : Base { // expected-note {{base class 'Base' declared here}}
  UninitBase() {} // expected-error {{constructor does not initialize base class 'Base' under profile 'std::init'}}
};

struct BaseInitParen : Base {
  BaseInitParen() : Base() {}
};

struct BaseInitBraces : Base {
  BaseInitBraces() : Base{} {}
};

// A base with a user-provided default constructor is trusted.
struct TrustedBaseCtor : WithCtor {
  TrustedBaseCtor() {}
};

// A base whose only indeterminate scalar is [[uninit]]-marked is trusted.
struct MarkedBaseSub : InnerMarked {
  MarkedBaseSub() {}
};

struct MixedBaseMember : Base { // expected-note {{base class 'Base' declared here}}
  int x; // expected-note {{member 'x' declared here}}
  MixedBaseMember() {}
  // expected-error@-1 {{constructor does not initialize member 'x' under profile 'std::init'}}
  // expected-error@-2 {{constructor does not initialize base class 'Base' under profile 'std::init'}}
};

struct MultipleBases : Base, Inner {
  // expected-note@-1 {{base class 'Base' declared here}}
  // expected-note@-2 {{base class 'Inner' declared here}}
  MultipleBases() {}
  // expected-error@-1 {{constructor does not initialize base class 'Base' under profile 'std::init'}}
  // expected-error@-2 {{constructor does not initialize base class 'Inner' under profile 'std::init'}}
};

struct OutOfLineBase : Base { // expected-note {{base class 'Base' declared here}}
  OutOfLineBase();
};
OutOfLineBase::OutOfLineBase() {} // expected-error {{constructor does not initialize base class 'Base' under profile 'std::init'}}

template <typename T>
struct TmplBase : T { // expected-note {{base class 'Base' declared here}}
  TmplBase() {} // expected-error {{constructor does not initialize base class 'Base' under profile 'std::init'}}
};
template struct TmplBase<Base>; // expected-note {{in instantiation of member function 'TmplBase<Base>::TmplBase' requested here}}

struct SuppressedBaseByRule : Base {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ctor_uninit_member")]] SuppressedBaseByRule() {}
};

// A virtual base is the most-derived constructor's responsibility, so leaving
// it uninitialized here is deferred: no diagnostic.
struct VirtualBase : virtual Base {
  VirtualBase() {}
};

// ============================================================
// Anonymous aggregate members
// ============================================================

// The leaves of an anonymous struct member initialize exactly like direct
// members (a written initializer for one is an indirect member-initializer),
// so a constructor must cover them the same way.
struct AnonStructMissing {
  struct {
    int x; // expected-note {{member 'x' declared here}}
  };
  AnonStructMissing() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};

struct AnonStructMemInit {
  struct {
    int x;
  };
  AnonStructMemInit() : x(1) {} // OK: indirect member-initializer covers the leaf
};

struct AnonStructNSDMI {
  struct {
    int x = 0;
  };
  AnonStructNSDMI() {} // OK: the leaf's default member initializer covers it
};

struct AnonStructMarked {
  struct {
    int x [[uninit]];
  };
  AnonStructMarked() {} // OK: the leaf's marker acknowledges it
};

struct AnonStructPartial {
  struct {
    int x;
    int y; // expected-note {{member 'y' declared here}}
  };
  AnonStructPartial() : x(1) {} // expected-error {{constructor does not initialize member 'y' under profile 'std::init'}}
};

struct AnonStructNested {
  struct {
    struct {
      int deep; // expected-note {{member 'deep' declared here}}
    };
  };
  AnonStructNested() {} // expected-error {{constructor does not initialize member 'deep' under profile 'std::init'}}
};

struct AnonStructSuppressed {
  struct {
    int x;
  };
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ctor_uninit_member")]] AnonStructSuppressed() {}
};

// An anonymous union needs one active member, not every member: a written
// leaf initializer or a leaf default member initializer satisfies it.
struct AnonUnionMissing {
  union { // expected-note {{anonymous union declared here}}
    int i;
    float f;
  };
  AnonUnionMissing() {} // expected-error {{constructor does not initialize any member of the anonymous union under profile 'std::init'}}
};

struct AnonUnionMemInit {
  union {
    int i;
    float f;
  };
  AnonUnionMemInit() : i(1) {} // OK: the written leaf is the active member
};

struct AnonUnionNSDMI {
  union {
    int i = 0;
    float f;
  };
  AnonUnionNSDMI() {} // OK: the leaf's default member initializer is the active member
};

// One written leaf of a struct variant activates the union; whether that
// variant is fully initialized is deliberately lenient (a missed diagnostic,
// never a false positive).
struct AnonUnionStructVariant {
  union {
    struct {
      int a;
      int b;
    };
    float f;
  };
  AnonUnionStructVariant() : a(1) {} // OK: lenient
};

struct AnonUnionSuppressed {
  union {
    int i;
    float f;
  };
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ctor_uninit_member")]] AnonUnionSuppressed() {}
};

// An anonymous union whose only members are unnamed bit-fields has nothing
// for the constructor to initialize: unnamed bit-fields are not members and
// no in-language initializer exists for them.
struct AnonUnionUnnamedBitFieldOnly {
  union {
    int : 4;
  };
  AnonUnionUnnamedBitFieldOnly() {} // OK: nothing to initialize
};

// A std::byte-only anonymous union mirrors the scalar std::byte exemption
// (paper §4.5): whichever member is active may be left uninitialized, so
// there is nothing to acknowledge -- matching the anonymous-struct control,
// which the per-member walk already exempts. A non-byte member alongside
// keeps the union flagged.
namespace std { enum class byte : unsigned char {}; }
struct AnonUnionAllByte {
  union {
    std::byte b;
  };
  AnonUnionAllByte() {} // OK: std::byte may be left uninitialized
};
struct AnonStructByteControl {
  struct {
    std::byte b;
  };
  AnonStructByteControl() {} // OK: already exempt via the member walk
};
struct AnonUnionMixedByte {
  union { // expected-note {{anonymous union declared here}}
    std::byte b;
    int i;
  };
  AnonUnionMixedByte() {} // expected-error {{constructor does not initialize any member of the anonymous union under profile 'std::init'}}
};

// Default member initializers on every leaf of an anonymous-struct variant
// activate it during default-initialization and initialize it completely;
// the static_assert below is the constant-evaluation proof.
struct AnonUnionAllNSDMIVariant {
  union {
    struct {
      int a = 1;
      int b = 2;
    };
    float f;
  };
  constexpr AnonUnionAllNSDMIVariant() {} // OK: the variant is active and complete
};
static_assert(AnonUnionAllNSDMIVariant().a == 1 &&
              AnonUnionAllNSDMIVariant().b == 2);

// A partial-NSDMI variant is activated all the same, but default-init
// leaves its other leaf indeterminate, so the constructor stays diagnosed.
struct AnonUnionPartialNSDMIVariant {
  union { // expected-note {{anonymous union declared here}}
    struct {
      int a = 1;
      int b;
    };
    float f;
  };
  AnonUnionPartialNSDMIVariant() {} // expected-error {{constructor does not initialize any member of the anonymous union under profile 'std::init'}}
};

// The activation is visible through nesting: an all-NSDMI anonymous struct
// variant of an anonymous union inside an anonymous struct.
struct AnonNestedNSDMIVariant {
  struct {
    union {
      struct {
        int a = 1;
      };
      float f;
    };
  };
  AnonNestedNSDMIVariant() {} // OK: the nested variant is active and complete
};
