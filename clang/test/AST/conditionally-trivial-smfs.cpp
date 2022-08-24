// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-linux -ast-dump=json %s | FileCheck %s --check-prefixes=CHECK,LIN
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-win32 -ast-dump=json %s | FileCheck %s

// This test validates that we compute correct AST properties of classes with
// conditionally trivial special member functions.

template <int N>
struct DefaultConstructorCheck {
  DefaultConstructorCheck() requires(N == 1) = default;
  DefaultConstructorCheck() requires(N == 2) = delete;
  DefaultConstructorCheck() requires(N == 3);
  DefaultConstructorCheck();
};


template struct DefaultConstructorCheck<1>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true,
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "isConstexpr": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userProvided": true
// CHECK-NEXT:          },

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,

template struct DefaultConstructorCheck<2>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true,
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "isConstexpr": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userProvided": true
// CHECK-NEXT:          },

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,


template struct DefaultConstructorCheck<3>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true,
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "isConstexpr": true,
// CHECK-NEXT:            "nonTrivial": true,
// CHECK-NEXT:            "userProvided": true
// CHECK-NEXT:          },

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,

template <int N>
struct CopyConstructorCheck {
  CopyConstructorCheck(const CopyConstructorCheck&) requires(N == 1) = default;
  CopyConstructorCheck(const CopyConstructorCheck&) requires(N == 2) = delete;
  CopyConstructorCheck(const CopyConstructorCheck&) requires(N == 3);
  CopyConstructorCheck(const CopyConstructorCheck&);
};


template struct CopyConstructorCheck<1>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },

// CHECK:               "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},

template struct CopyConstructorCheck<2>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },

// CHECK:               "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},

template struct CopyConstructorCheck<3>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "nonTrivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },

// CHECK:               "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "moveAssign": {},

template <int N>
struct MoveConstructorCheck {
  MoveConstructorCheck(MoveConstructorCheck&&) requires(N == 1) = default;
  MoveConstructorCheck(MoveConstructorCheck&&) requires(N == 2) = delete;
  MoveConstructorCheck(MoveConstructorCheck&&) requires(N == 3);
  MoveConstructorCheck(MoveConstructorCheck&&);
};


template struct MoveConstructorCheck<1>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// LIN-NEXT:            "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          }

template struct MoveConstructorCheck<2>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          }

template struct MoveConstructorCheck<3>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "hasUserDeclaredConstructor": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "nonTrivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          }

template <int N>
struct CopyAssignmentCheck {
  CopyAssignmentCheck& operator=(const CopyAssignmentCheck&) requires(N == 1) = default;
  CopyAssignmentCheck& operator=(const CopyAssignmentCheck&) requires(N == 2) = delete;
  CopyAssignmentCheck& operator=(const CopyAssignmentCheck&) requires(N == 3);
  CopyAssignmentCheck& operator=(const CopyAssignmentCheck&);
};


template struct CopyAssignmentCheck<1>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT           "copyAssign": {
// CHECK-NEXT             "hasConstParam": true,
// CHECK-NEXT             "implicitHasConstParam": true,
// CHECK-NEXT             "trivial": true,
// CHECK-NEXT             "userDeclared": true
// CHECK-NEXT           },

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT           "isAggregate": true,
// CHECK-NEXT           "isEmpty": true,
// CHECK-NEXT           "isLiteral": true,
// CHECK-NEXT           "isStandardLayout": true,
// CHECK-NEXT           "isTrivial": true,
// CHECK-NEXT           "isTriviallyCopyable": true,
// CHECK-NEXT           "moveAssign": {},

template struct CopyAssignmentCheck<2>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT           "copyAssign": {
// CHECK-NEXT             "hasConstParam": true,
// CHECK-NEXT             "implicitHasConstParam": true,
// CHECK-NEXT             "trivial": true,
// CHECK-NEXT             "userDeclared": true
// CHECK-NEXT           },

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT           "isAggregate": true,
// CHECK-NEXT           "isEmpty": true,
// CHECK-NEXT           "isLiteral": true,
// CHECK-NEXT           "isStandardLayout": true,
// CHECK-NEXT           "isTrivial": true,
// CHECK-NEXT           "isTriviallyCopyable": true,
// CHECK-NEXT           "moveAssign": {},

template struct CopyAssignmentCheck<3>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT           "copyAssign": {
// CHECK-NEXT             "hasConstParam": true,
// CHECK-NEXT             "implicitHasConstParam": true,
// CHECK-NEXT             "trivial": true,
// CHECK-NEXT             "userDeclared": true
// CHECK-NEXT           },

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT           "isAggregate": true,
// CHECK-NEXT           "isEmpty": true,
// CHECK-NEXT           "isLiteral": true,
// CHECK-NEXT           "isStandardLayout": true,
// CHECK-NEXT           "moveAssign": {},

template <int N>
struct MoveAssignmentCheck {
  MoveAssignmentCheck& operator=(MoveAssignmentCheck&&) requires(N == 1) = default;
  MoveAssignmentCheck& operator=(MoveAssignmentCheck&&) requires(N == 2) = delete;
  MoveAssignmentCheck& operator=(MoveAssignmentCheck&&) requires(N == 3);
  MoveAssignmentCheck& operator=(MoveAssignmentCheck&&);
};


template struct MoveAssignmentCheck<1>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },

template struct MoveAssignmentCheck<2>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },

template struct MoveAssignmentCheck<3>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "moveAssign": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "nonTrivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },
