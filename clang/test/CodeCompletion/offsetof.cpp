struct S {
  int field;
  int other;
  void method();
};

struct Inner {
  int leaf;
  int otherLeaf;
  void method();
};

struct Outer {
  Inner inner;
  Inner array[2];
};

struct Base {
  int inherited;
};

struct Derived : Base {
  int direct;
};

struct RefOuter {
  Inner &ref;
};

struct WithAnon {
  int outer;
  union {
    int anonInt;
    Inner anonInner;
  };
};

struct ShadowBase {
  Inner shadowed;
};

struct ShadowDerived : ShadowBase {
  using ShadowBase::shadowed;
};

struct WithBitField {
  int regular;
  int bit : 4;
  int : 4;
  int otherRegular;
};

struct WithAnonBitField {
  int outer;
  struct {
    int anonRegular;
    int anonBit : 4;
  };
};

#define offsetof(type, member) __builtin_offsetof(type, member)

// Cursor immediately after the comma: empty designator path.
int empty = __builtin_offsetof(S, field);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):35 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-S --implicit-check-not=method %s

// Cursor after a dot: completion in the nested record's type.
int after_dot = __builtin_offsetof(Outer, inner.leaf);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):49 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-INNER --implicit-check-not=method %s

// Cursor after a dot following an array subscript.
int array = __builtin_offsetof(Outer, array[0].leaf);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):48 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-INNER --implicit-check-not=method %s

// Inherited fields participate in offsetof completion.
int inherited = __builtin_offsetof(Derived, inherited);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):45 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-DERIVED %s

// Reference field: dereferenced before continuing the path.
int ref = __builtin_offsetof(RefOuter, ref.leaf);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):44 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-INNER --implicit-check-not=method %s

// Empty path on a record with anonymous-member indirect fields.
int anon_empty = __builtin_offsetof(WithAnon, anonInt);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):47 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-ANON %s

// Designator that starts with an indirect field from an anonymous member.
int anon_nested = __builtin_offsetof(WithAnon, anonInner.leaf);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):58 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-INNER --implicit-check-not=method %s

// Field exposed via `using Base::...` resolves through its UsingShadowDecl.
int shadowed = __builtin_offsetof(ShadowDerived, shadowed.leaf);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):59 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-INNER --implicit-check-not=method %s

// Macro form expands to __builtin_offsetof with the same completion behavior.
int macro = offsetof(S, field);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):25 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-S --implicit-check-not=method %s

// Bit-fields are not valid as the offsetof member designator
// (err_offsetof_bitfield), so they should not be offered as completions.
// Unnamed bit-fields have no identifier and are filtered out of lookup
// independently.
int bf = __builtin_offsetof(WithBitField, regular);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):43 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-BF --implicit-check-not=bit %s

// Bit-fields exposed via an anonymous struct's IndirectFieldDecl are also
// filtered: the leaf field is what offsetof would name.
int abf = __builtin_offsetof(WithAnonBitField, outer);
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):48 %s -o - -std=c++17 | FileCheck -check-prefix=CHECK-ABF --implicit-check-not=anonBit %s

// CHECK-S-DAG: COMPLETION: field : [#int#]field
// CHECK-S-DAG: COMPLETION: other : [#int#]other

// CHECK-INNER-DAG: COMPLETION: leaf : [#int#]leaf
// CHECK-INNER-DAG: COMPLETION: otherLeaf : [#int#]otherLeaf

// CHECK-DERIVED-DAG: COMPLETION: direct : [#int#]direct
// CHECK-DERIVED-DAG: COMPLETION: inherited (InBase) : [#int#]inherited

// CHECK-ANON-DAG: COMPLETION: anonInner : [#Inner#]anonInner
// CHECK-ANON-DAG: COMPLETION: anonInt : [#int#]anonInt
// CHECK-ANON-DAG: COMPLETION: outer : [#int#]outer

// CHECK-BF-DAG: COMPLETION: regular : [#int#]regular
// CHECK-BF-DAG: COMPLETION: otherRegular : [#int#]otherRegular

// CHECK-ABF-DAG: COMPLETION: outer : [#int#]outer
// CHECK-ABF-DAG: COMPLETION: anonRegular : [#int#]anonRegular
