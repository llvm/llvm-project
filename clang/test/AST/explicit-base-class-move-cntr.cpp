// RUN: %clang_cc1 -ast-dump=json %s | FileCheck -strict-whitespace %s

struct ExplicitBase {
  explicit ExplicitBase(const char *) { }
  ExplicitBase(const ExplicitBase &) {}
  ExplicitBase(ExplicitBase &&) {}
  ExplicitBase &operator=(const ExplicitBase &) { return *this; }
  ExplicitBase &operator=(ExplicitBase &&) { return *this; }
  ~ExplicitBase() { }
};

struct Derived1 : ExplicitBase {};

Derived1 makeDerived1() {
// CHECK:  "kind": "FunctionDecl",
// CHECK:  "name": "makeDerived1",

// CHECK:    "kind": "CompoundStmt",

// CHECK:      "kind": "ReturnStmt",
// CHECK:        "kind": "ExprWithCleanups",
// CHECK:        "type": {
// CHECK-NEXT:     "qualType": "Derived1"
// CHECK-NEXT:   },

// CHECK:          "kind": "CXXFunctionalCastExpr",
// CHECK:          "type": {
// CHECK-NEXT:       "qualType": "Derived1"
// CHECK-NEXT:     },
// CHECK-NEXT:     "valueCategory": "prvalue",
// CHECK-NEXT:     "castKind": "NoOp",

// CHECK:            "kind": "CXXBindTemporaryExpr",
// CHECK:            "type": {
// CHECK-NEXT:         "qualType": "Derived1"
// CHECK-NEXT:       },
// CHECK-NEXT:       "valueCategory": "prvalue",

// CHECK:              "kind": "InitListExpr",
// CHECK:              "type": {
// CHECK-NEXT:           "qualType": "Derived1"
// CHECK-NEXT:         },
// CHECK-NEXT:         "valueCategory": "prvalue",

// CHECK:                "kind": "CXXConstructExpr",
// CHECK:                "type": {
// CHECK-NEXT:             "qualType": "ExplicitBase"
// CHECK-NEXT:           },
// CHECK-NEXT:           "valueCategory": "prvalue",
// CHECK-NEXT:           "ctorType": {
// CHECK-NEXT:             "qualType": "void (ExplicitBase &&)"
// CHECK-NEXT:           },
// CHECK-NEXT:           "hadMultipleCandidates": true,
// CHECK-NEXT:           "constructionKind": "non-virtual base",

// CHECK:                  "kind": "MaterializeTemporaryExpr",
// CHECK:                  "type": {
// CHECK-NEXT:               "qualType": "ExplicitBase"
// CHECK-NEXT:             },
// CHECK-NEXT:             "valueCategory": "xvalue",
// CHECK-NEXT:             "storageDuration": "full expression",

// CHECK:                    "kind": "CXXBindTemporaryExpr",
// CHECK:                    "type": {
// CHECK-NEXT:                 "qualType": "ExplicitBase"
// CHECK-NEXT:               },
// CHECK-NEXT:               "valueCategory": "prvalue",

// CHECK:                      "kind": "CXXTemporaryObjectExpr",
// CHECK:                      "type": {
// CHECK-NEXT:                   "qualType": "ExplicitBase"
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "valueCategory": "prvalue",
// CHECK-NEXT:                 "ctorType": {
// CHECK-NEXT:                   "qualType": "void (const char *)"
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "list": true,
// CHECK-NEXT:                 "hadMultipleCandidates": true,
// CHECK-NEXT:                 "constructionKind": "complete",

// CHECK:                        "kind": "ImplicitCastExpr",
// CHECK:                        "type": {
// CHECK-NEXT:                     "qualType": "const char *"
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "valueCategory": "prvalue",
// CHECK-NEXT:                   "castKind": "ArrayToPointerDecay",

// CHECK:                          "kind": "StringLiteral",
// CHECK:                          "type": {
// CHECK-NEXT:                       "qualType": "const char[10]"
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "valueCategory": "lvalue",
// CHECK-NEXT:                     "value": "\"Move Ctor\""
  return Derived1{ExplicitBase{"Move Ctor"}};
}

struct ImplicitBase {
  ImplicitBase(const char *) { }
  ImplicitBase(const ImplicitBase &) {}
  ImplicitBase(ImplicitBase &&) {}
  ImplicitBase &operator=(const ImplicitBase &) { return *this; }
  ImplicitBase &operator=(ImplicitBase &&) { return *this; }
  ~ImplicitBase() { }
};

struct Derived2 : ImplicitBase {};

Derived2 makeDerived2() {
// CHECK:  "kind": "FunctionDecl",
// CHECK:  "name": "makeDerived2",

// CHECK:    "kind": "CompoundStmt",

// CHECK:      "kind": "ReturnStmt",

// CHECK:        "kind": "ExprWithCleanups",
// CHECK:        "type": {
// CHECK-NEXT:     "qualType": "Derived2"
// CHECK-NEXT:   },
// CHECK-NEXT:   "valueCategory": "prvalue",
// CHECK-NEXT:   "cleanupsHaveSideEffects": true,

// CHECK:          "kind": "CXXFunctionalCastExpr",
// CHECK:          "type": {
// CHECK-NEXT:       "qualType": "Derived2"
// CHECK-NEXT:     },
// CHECK-NEXT:     "valueCategory": "prvalue",
// CHECK-NEXT:     "castKind": "NoOp",

// CHECK:            "kind": "CXXBindTemporaryExpr",
// CHECK:            "type": {
// CHECK-NEXT:         "qualType": "Derived2"
// CHECK-NEXT:       },
// CHECK-NEXT:       "valueCategory": "prvalue",

// CHECK:              "kind": "InitListExpr",
// CHECK:              "type": {
// CHECK-NEXT:           "qualType": "Derived2"
// CHECK-NEXT:         },
// CHECK-NEXT:         "valueCategory": "prvalue",

// CHECK:                "kind": "CXXConstructExpr",
// CHECK:                "type": {
// CHECK-NEXT:             "qualType": "ImplicitBase"
// CHECK-NEXT:           },
// CHECK-NEXT:           "valueCategory": "prvalue",
// CHECK-NEXT:           "ctorType": {
// CHECK-NEXT:             "qualType": "void (const char *)"
// CHECK-NEXT:           },
// CHECK-NEXT:           "list": true,
// CHECK-NEXT:           "hadMultipleCandidates": true,
// CHECK-NEXT:           "constructionKind": "non-virtual base",

// CHECK:                  "kind": "ImplicitCastExpr",
// CHECK:                  "type": {
// CHECK-NEXT:               "qualType": "const char *"
// CHECK-NEXT:             },
// CHECK-NEXT:             "valueCategory": "prvalue",
// CHECK-NEXT:             "castKind": "ArrayToPointerDecay",

// CHECK:                    "kind": "StringLiteral",
// CHECK:                    "type": {
// CHECK-NEXT:                 "qualType": "const char[8]"
// CHECK-NEXT:               },
// CHECK-NEXT:               "valueCategory": "lvalue",
// CHECK-NEXT:               "value": "\"No Ctor\""
  return Derived2{{"No Ctor"}};
}

// NOTE: CHECK lines have been autogenerated by gen_ast_dump_json_test.py
// using --filters=FunctionDecl,CompoundStmt,ReturnStmt,MaterializeTemporaryExpr,CXXBindTemporaryExpr,CXXTemporaryObjectExpr,ImplicitCastExpr,StringLiteralStringLiteral
