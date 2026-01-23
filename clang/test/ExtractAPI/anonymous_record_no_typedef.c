// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -isystem %S -fretain-comments-from-system-headers \
// RUN:   -x c-header %s -o %t/output-c.symbols.json -verify
//
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -isystem %S -fretain-comments-from-system-headers \
// RUN:   -x c++-header %s -o %t/output-cxx.symbols.json -verify

// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix GLOBAL
// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix PREFIX
// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix CONTENT
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix GLOBAL
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix PREFIX
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix CONTENT
/// A global variable with an anonymous struct type.
struct { char *prefix; char *content; } global;
// GLOBAL-LABEL: "!testLabel": "c:@global"
// GLOBAL:      "declarationFragments": [
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "keyword",
// GLOBAL-NEXT:     "spelling": "struct"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "text",
// GLOBAL-NEXT:     "spelling": " { ... } "
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "identifier",
// GLOBAL-NEXT:     "spelling": "global"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "text",
// GLOBAL-NEXT:     "spelling": ";"
// GLOBAL-NEXT:   }
// GLOBAL-NEXT: ],
// GLOBAL: "text": "A global variable with an anonymous struct type."
// GLOBAL:     "kind": {
// GLOBAL-NEXT:  "displayName": "Global Variable",
// GLOBAL-NEXT:  "identifier": "c{{(\+\+)?}}.var"
// GLOBAL:       "title": "global"
// GLOBAL:     "pathComponents": [
// GLOBAL-NEXT:  "global"
// GLOBAL-NEXT:]

// PREFIX: "!testRelLabel": "memberOf $ c:@S@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@prefix $ c:@global"
// PREFIX-LABEL: "!testLabel": "c:@S@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@prefix"
// PREFIX: "title": "prefix"
// PREFIX:      "pathComponents": [
// PREFIX-NEXT:   "global",
// PREFIX-NEXT:   "prefix"
// PREFIX-NEXT: ]

// CONTENT: "!testRelLabel": "memberOf $ c:@S@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@content $ c:@global"
// CONTENT-LABEL: "!testLabel": "c:@S@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@content"
// CONTENT: "title": "content"
// CONTENT:      "pathComponents": [
// CONTENT-NEXT:   "global",
// CONTENT-NEXT:   "content"
// CONTENT-NEXT: ]

/// A Vehicle
struct Vehicle {
    // RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix TYPE
    // RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix BICYCLE
    // RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix CAR
    // RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix TYPE
    // RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix BICYCLE
    // RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix CAR
    /// The type of vehicle.
    enum {
        Bicycle,
        Car
    } type;
    // TYPE-LABEL: "!testLabel": "c:@S@Vehicle@FI@type"
    // TYPE:      "declarationFragments": [
    // TYPE-NEXT:   {
    // TYPE-NEXT:     "kind": "keyword",
    // TYPE-NEXT:     "spelling": "enum"
    // TYPE-NEXT:   },
    // TYPE-NEXT:   {
    // TYPE-NEXT:     "kind": "text",
    // TYPE-NEXT:     "spelling": " { ... } "
    // TYPE-NEXT:   },
    // TYPE-NEXT:   {
    // TYPE-NEXT:     "kind": "identifier",
    // TYPE-NEXT:     "spelling": "type"
    // TYPE-NEXT:   },
    // TYPE-NEXT:   {
    // TYPE-NEXT:     "kind": "text",
    // TYPE-NEXT:     "spelling": ";"
    // TYPE-NEXT:   }
    // TYPE-NEXT: ],
    // TYPE: "text": "The type of vehicle."
    // TYPE: "title": "type"

    // BICYCLE-LABEL: "!testLabel": "c:@S@Vehicle@E@anonymous_record_no_typedef.c@{{[0-9]+}}@Bicycle"
    // BICYCLE: "title": "Bicycle"
    // BICYCLE:      "pathComponents": [
    // BICYCLE-NEXT:   "Bicycle"
    // BICYCLE-NEXT: ]

    // CAR-LABEL: "!testLabel": "c:@S@Vehicle@E@anonymous_record_no_typedef.c@{{[0-9]+}}@Car"
    // CAR: "title": "Car"
    // CAR:      "pathComponents": [
    // CAR-NEXT:   "Car"
    // CAR-NEXT: ]

    // RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix INFORMATION
    // RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix WHEELS
    // RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix NAME
    // RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix INFORMATION
    // RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix WHEELS
    // RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix NAME
    /// The information about the vehicle.
    union {
        int wheels;
        char *name;
    } information;
    // INFORMATION-LABEL: "!testLabel": "c:@S@Vehicle@FI@information"
    // INFORMATION:      "declarationFragments": [
    // INFORMATION-NEXT:   {
    // INFORMATION-NEXT:     "kind": "keyword",
    // INFORMATION-NEXT:     "spelling": "union"
    // INFORMATION-NEXT:   },
    // INFORMATION-NEXT:   {
    // INFORMATION-NEXT:     "kind": "text",
    // INFORMATION-NEXT:     "spelling": " { ... } "
    // INFORMATION-NEXT:   },
    // INFORMATION-NEXT:   {
    // INFORMATION-NEXT:     "kind": "identifier",
    // INFORMATION-NEXT:     "spelling": "information"
    // INFORMATION-NEXT:   },
    // INFORMATION-NEXT:   {
    // INFORMATION-NEXT:     "kind": "text",
    // INFORMATION-NEXT:     "spelling": ";"
    // INFORMATION-NEXT:   }
    // INFORMATION-NEXT: ],
    // INFORMATION: "text": "The information about the vehicle."
    // INFORMATION: "title": "information"

    // WHEELS: "!testRelLabel": "memberOf $ c:@S@Vehicle@U@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@wheels $ c:@S@Vehicle@FI@information"
    // WHEELS-LABEL: "!testLabel": "c:@S@Vehicle@U@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@wheels"
    // WHEELS: "title": "wheels"
    // WHEELS:      "pathComponents": [
    // WHEELS-NEXT:   "Vehicle",
    // WHEELS-NEXT:   "information",
    // WHEELS-NEXT:   "wheels"
    // WHEELS-NEXT: ]

    // NAME: "!testRelLabel": "memberOf $ c:@S@Vehicle@U@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@name $ c:@S@Vehicle@FI@information"
    // NAME-LABEL: "!testLabel": "c:@S@Vehicle@U@anonymous_record_no_typedef.c@{{[0-9]+}}@FI@name"
    // NAME: "title": "name"
    // NAME:      "pathComponents": [
    // NAME-NEXT:   "Vehicle",
    // NAME-NEXT:   "information",
    // NAME-NEXT:   "name"
    // NAME-NEXT: ]
};

// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix GLOBALCASE
// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix GLOBALOTHERCASE
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix GLOBALCASE
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix GLOBALOTHERCASE
enum {
  GlobalCase,
  GlobalOtherCase
};
// GLOBALCASE-LABEL: "!testLabel": "c:@Ea@GlobalCase@GlobalCase"
// GLOBALCASE: "title": "GlobalCase"
// GLOBALCASE:      "pathComponents": [
// GLOBALCASE-NEXT:   "GlobalCase"
// GLOBALCASE-NEXT: ]

// GLOBALOTHERCASE-LABEL: "!testLabel": "c:@Ea@GlobalCase@GlobalOtherCase"
// GLOBALOTHERCASE: "title": "GlobalOtherCase"
// GLOBALOTHERCASE:      "pathComponents": [
// GLOBALOTHERCASE-NEXT:   "GlobalOtherCase"
// GLOBALOTHERCASE-NEXT: ]

// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix VEC
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix VEC
union Vector {
  struct {
    float X;
    float Y;
  };
  float Data[2];
};
// VEC-DAG: "!testRelLabel": "memberOf $ c:@U@Vector@FI@Data $ c:@U@Vector"
// VEC-DAG: "!testRelLabel": "memberOf $ c:@U@Vector@Sa@FI@X $ c:@U@Vector"
// VEC-DAG: "!testRelLabel": "memberOf $ c:@U@Vector@Sa@FI@Y $ c:@U@Vector"

// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix MYSTRUCT
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix MYSTRUCT
// RUN: FileCheck %s --input-file %t/output-c.symbols.json --check-prefix COUNTS
// RUN: FileCheck %s --input-file %t/output-cxx.symbols.json --check-prefix COUNTS
struct MyStruct {
    struct {
        int count;
    } counts[1];
};
// MYSTRUCT-NOT: "spelling": ""
// MYSTRUCT-NOT: "title": ""

// COUNTS-LABEL: "!testLabel": "c:@S@MyStruct@FI@counts"
// COUNTS:      "declarationFragments": [
// COUNTS-NEXT:   {
// COUNTS-NEXT:     "kind": "keyword",
// COUNTS-NEXT:     "spelling": "struct"
// COUNTS-NEXT:   },
// COUNTS-NEXT:   {
// COUNTS-NEXT:     "kind": "text",
// COUNTS-NEXT:     "spelling": " { ... } "
// COUNTS-NEXT:   },
// COUNTS-NEXT:   {
// COUNTS-NEXT:     "kind": "identifier",
// COUNTS-NEXT:     "spelling": "counts"
// COUNTS-NEXT:   },
// COUNTS-NEXT:   {
// COUNTS-NEXT:     "kind": "text",
// COUNTS-NEXT:     "spelling": "["
// COUNTS-NEXT:   },
// COUNTS-NEXT:   {
// COUNTS-NEXT:     "kind": "number",
// COUNTS-NEXT:     "spelling": "1"
// COUNTS-NEXT:   },
// COUNTS-NEXT:   {
// COUNTS-NEXT:     "kind": "text",
// COUNTS-NEXT:     "spelling": "];"
// COUNTS-NEXT:   }
// COUNTS-NEXT: ],

// expected-no-diagnostics
