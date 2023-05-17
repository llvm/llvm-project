// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-linux -ast-dump=json %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-win32 -ast-dump=json %s | FileCheck %s -check-prefixes=CHECK,WIN32

// This test validates that we compute correct AST properties of classes after choosing
// their destructor when doing destructor overload resolution with concepts.

template <int N>
struct A {
  ~A() requires(N == 1) = default;
  ~A() requires(N == 2) = delete;
  ~A() requires(N == 3);
  constexpr ~A() requires(N == 4);

private:
  ~A() requires(N == 5) = default;
};


template struct A<1>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },
// CHECK-NEXT:          "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {}

template struct A<2>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "dtor": {
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },
// CHECK-NEXT:          "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {}

template struct A<3>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// WIN32-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "dtor": {
// CHECK-NEXT:            "nonTrivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },
// CHECK-NEXT:          "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {}

template struct A<4>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// WIN32-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "dtor": {
// CHECK-NEXT:            "nonTrivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },
// CHECK-NEXT:          "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {}

template struct A<5>;
// CHECK:             "kind": "ClassTemplateSpecializationDecl",
// CHECK:             "definitionData": {
// CHECK-NEXT:          "canConstDefaultInit": true,
// CHECK-NEXT:          "canPassInRegisters": true,
// CHECK-NEXT:          "copyAssign": {

// CHECK:               "dtor": {
// CHECK-NEXT:            "trivial": true,
// CHECK-NEXT:            "userDeclared": true
// CHECK-NEXT:          },
// CHECK-NEXT:          "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:          "isAggregate": true,
// CHECK-NEXT:          "isEmpty": true,
// CHECK-NEXT:          "isLiteral": true,
// CHECK-NEXT:          "isStandardLayout": true,
// CHECK-NEXT:          "isTrivial": true,
// CHECK-NEXT:          "isTriviallyCopyable": true,
// CHECK-NEXT:          "moveAssign": {},
// CHECK-NEXT:          "moveCtor": {}

