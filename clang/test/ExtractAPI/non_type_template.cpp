// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -std=c++17 -x c++-header %s -o %t/output.symbols.json -verify

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix FOO
template <typename T, int N = 4> class Foo { };
// FOO-LABEL: "!testLabel": "c:@ST>2#T#NI@Foo"
// FOO:      "declarationFragments": [
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "keyword",
// FOO-NEXT:     "spelling": "template"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": " <"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "keyword",
// FOO-NEXT:     "spelling": "typename"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": " "
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "genericParameter",
// FOO-NEXT:     "spelling": "T"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": ", "
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "typeIdentifier",
// FOO-NEXT:     "preciseIdentifier": "c:I",
// FOO-NEXT:     "spelling": "int"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": " "
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "genericParameter",
// FOO-NEXT:     "spelling": "N"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": " = 4> "
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "keyword",
// FOO-NEXT:     "spelling": "class"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": " "
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "identifier",
// FOO-NEXT:     "spelling": "Foo"
// FOO-NEXT:   },
// FOO-NEXT:   {
// FOO-NEXT:     "kind": "text",
// FOO-NEXT:     "spelling": ";"
// FOO-NEXT:   }
// FOO-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix FOO-SPEC
template <typename T> class Foo <T, 4> { };
// FOO-SPEC-LABEL: "!testLabel": "c:@SP>1#T@Foo>#t0.0#VI4"
// FOO-SPEC:      "declarationFragments": [
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "keyword",
// FOO-SPEC-NEXT:     "spelling": "template"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "text",
// FOO-SPEC-NEXT:     "spelling": " <"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "keyword",
// FOO-SPEC-NEXT:     "spelling": "typename"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "text",
// FOO-SPEC-NEXT:     "spelling": " "
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "genericParameter",
// FOO-SPEC-NEXT:     "spelling": "T"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "text",
// FOO-SPEC-NEXT:     "spelling": "> "
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "keyword",
// FOO-SPEC-NEXT:     "spelling": "class"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "text",
// FOO-SPEC-NEXT:     "spelling": " "
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "identifier",
// FOO-SPEC-NEXT:     "spelling": "Foo"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "text",
// FOO-SPEC-NEXT:     "spelling": "<"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "typeIdentifier",
// FOO-SPEC-NEXT:     "preciseIdentifier": "c:t0.0",
// FOO-SPEC-NEXT:     "spelling": "T"
// FOO-SPEC-NEXT:   },
// FOO-SPEC-NEXT:   {
// FOO-SPEC-NEXT:     "kind": "text",
// FOO-SPEC-NEXT:     "spelling": ", 4>;"
// FOO-SPEC-NEXT:   }
// FOO-SPEC-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix NEST
template <template <template <typename> typename> class... Bs> class NestedTemplateTemplateParamPack{ };
// NEST-LABEL: "!testLabel": "c:@ST>1#pt>1#t>1#T@NestedTemplateTemplateParamPack"
// NEST:      "declarationFragments": [
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "template"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": " <"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "template"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": " <"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "template"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": " <"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "typename"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": "> "
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "typename"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": "> "
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "class"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": "... "
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "genericParameter",
// NEST-NEXT:     "spelling": "Bs"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": "> "
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "keyword",
// NEST-NEXT:     "spelling": "class"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": " "
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "identifier",
// NEST-NEXT:     "spelling": "NestedTemplateTemplateParamPack"
// NEST-NEXT:   },
// NEST-NEXT:   {
// NEST-NEXT:     "kind": "text",
// NEST-NEXT:     "spelling": ";"
// NEST-NEXT:   }
// NEST-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix BAR
template <template <typename> typename T = Foo> struct Bar { };
// BAR-LABEL: "!testLabel": "c:@ST>1#t>1#T@Bar"
// BAR:      "declarationFragments": [
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "keyword",
// BAR-NEXT:     "spelling": "template"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": " <"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "keyword",
// BAR-NEXT:     "spelling": "template"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": " <"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "keyword",
// BAR-NEXT:     "spelling": "typename"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": "> "
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "keyword",
// BAR-NEXT:     "spelling": "typename"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": " "
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "genericParameter",
// BAR-NEXT:     "spelling": "T"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": " = "
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "typeIdentifier",
// BAR-NEXT:     "preciseIdentifier": "c:@ST>2#T#NI@Foo",
// BAR-NEXT:     "spelling": "Foo"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": "> "
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "keyword",
// BAR-NEXT:     "spelling": "struct"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": " "
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "identifier",
// BAR-NEXT:     "spelling": "Bar"
// BAR-NEXT:   },
// BAR-NEXT:   {
// BAR-NEXT:     "kind": "text",
// BAR-NEXT:     "spelling": ";"
// BAR-NEXT:   }
// BAR-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix VAR
NestedTemplateTemplateParamPack<Bar, Bar> var;
// VAR-LABEL: "!testLabel": "c:@var"
// VAR:      "declarationFragments": [
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "typeIdentifier",
// VAR-NEXT:     "preciseIdentifier": "c:@ST>1#pt>1#t>1#T@NestedTemplateTemplateParamPack",
// VAR-NEXT:     "spelling": "NestedTemplateTemplateParamPack"
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "text",
// VAR-NEXT:     "spelling": "<"
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "typeIdentifier",
// VAR-NEXT:     "preciseIdentifier": "c:@ST>1#t>1#T@Bar",
// VAR-NEXT:     "spelling": "Bar"
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "text",
// VAR-NEXT:     "spelling": ", "
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "typeIdentifier",
// VAR-NEXT:     "preciseIdentifier": "c:@ST>1#t>1#T@Bar",
// VAR-NEXT:     "spelling": "Bar"
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "text",
// VAR-NEXT:     "spelling": "> "
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "identifier",
// VAR-NEXT:     "spelling": "var"
// VAR-NEXT:   },
// VAR-NEXT:   {
// VAR-NEXT:     "kind": "text",
// VAR-NEXT:     "spelling": ";"
// VAR-NEXT:   }
// VAR-NEXT: ]

template <typename T>
class TypeContainer {
  public:
    // RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix TYPE
    typedef Foo<T> Type;
// TYPE-LABEL: "!testLabel": "c:non_type_template.cpp@ST>1#T@TypeContainer@T@Type",
// TYPE:      "declarationFragments": [
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "keyword",
// TYPE-NEXT:     "spelling": "typedef"
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "text",
// TYPE-NEXT:     "spelling": " "
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "typeIdentifier",
// TYPE-NEXT:     "preciseIdentifier": "c:@ST>2#T#NI@Foo",
// TYPE-NEXT:     "spelling": "Foo"
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "text",
// TYPE-NEXT:     "spelling": "<"
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "typeIdentifier",
// TYPE-NEXT:     "preciseIdentifier": "c:t0.0",
// TYPE-NEXT:     "spelling": "T"
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "text",
// TYPE-NEXT:     "spelling": "> "
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "identifier",
// TYPE-NEXT:     "spelling": "Type"
// TYPE-NEXT:   },
// TYPE-NEXT:   {
// TYPE-NEXT:     "kind": "text",
// TYPE-NEXT:     "spelling": ";"
// TYPE-NEXT:   }
// TYPE-NEXT: ]
};

// expected-no-diagnostics
