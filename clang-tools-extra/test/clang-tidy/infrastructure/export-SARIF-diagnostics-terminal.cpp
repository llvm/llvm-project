// RUN: grep -Ev "// *[A-Z-]+:" %s > %t-input.cpp
// RUN: not clang-tidy %t-input.cpp -checks='-*,google-explicit-constructor,clang-diagnostic-missing-prototypes,clang-diagnostic-zero-length-array' --warnings-as-errors='clang-diagnostic-missing-prototypes,google-explicit-constructor' -sarif-export - -- -Wmissing-prototypes -Wzero-length-array 2>/dev/null | FileCheck %s
#define X(n) void n ## n() {}
X(f)
int a[-1];
int b[0];

void test(x);
struct Foo {
  member;
  Foo(int) {}
};

//CHECK: {{^{$}}
//CHECK-NEXT:   "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json",
//CHECK-NEXT:   "runs": [
//CHECK-NEXT:     {
//CHECK-NEXT:       "artifacts": [
//CHECK-NEXT:         {
//CHECK-NEXT:           "length": {{[0-9]+}},
//CHECK-NEXT:           "location": {
//CHECK-NEXT:             "index": 0,
//CHECK-NEXT:             "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:           },
//CHECK-NEXT:           "mimeType": "text/plain",
//CHECK-NEXT:           "roles": [
//CHECK-NEXT:             "resultFile"
//CHECK-NEXT:           ]
//CHECK-NEXT:         }
//CHECK-NEXT:       ],
//CHECK-NEXT:       "columnKind": "unicodeCodePoints",
//CHECK-NEXT:       "results": [
//CHECK-NEXT:         {
//CHECK-NEXT:           "level": "error",
//CHECK-NEXT:           "locations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 2,
//CHECK-NEXT:                   "endLine": 2,
//CHECK-NEXT:                   "startColumn": 1,
//CHECK-NEXT:                   "startLine": 2
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "message": {
//CHECK-NEXT:             "text": "no previous prototype for function 'ff'"
//CHECK-NEXT:           },
//CHECK-NEXT:           "relatedLocations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "message": {
//CHECK-NEXT:                 "text": "expanded from macro 'X'"
//CHECK-NEXT:               },
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 20,
//CHECK-NEXT:                   "endLine": 1,
//CHECK-NEXT:                   "startColumn": 19,
//CHECK-NEXT:                   "startLine": 1
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             },
//CHECK-NEXT:             {
//CHECK-NEXT:               "message": {
//CHECK-NEXT:                 "text": "declare 'static' if the function is not intended to be used outside of this translation unit"
//CHECK-NEXT:               },
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 2,
//CHECK-NEXT:                   "endLine": 2,
//CHECK-NEXT:                   "startColumn": 1,
//CHECK-NEXT:                   "startLine": 2
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             },
//CHECK-NEXT:             {
//CHECK-NEXT:               "message": {
//CHECK-NEXT:                 "text": "expanded from macro 'X'"
//CHECK-NEXT:               },
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 15,
//CHECK-NEXT:                   "endLine": 1,
//CHECK-NEXT:                   "startColumn": 14,
//CHECK-NEXT:                   "startLine": 1
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "ruleId": "clang-diagnostic-missing-prototypes",
//CHECK-NEXT:           "ruleIndex": 0
//CHECK-NEXT:         },
//CHECK-NEXT:         {
//CHECK-NEXT:           "level": "error",
//CHECK-NEXT:           "locations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 9,
//CHECK-NEXT:                   "endLine": 3,
//CHECK-NEXT:                   "startColumn": 7,
//CHECK-NEXT:                   "startLine": 3
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "message": {
//CHECK-NEXT:             "text": "'a' declared as an array with a negative size"
//CHECK-NEXT:           },
//CHECK-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-NEXT:           "ruleIndex": 1
//CHECK-NEXT:         },
//CHECK-NEXT:         {
//CHECK-NEXT:           "level": "warning",
//CHECK-NEXT:           "locations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 8,
//CHECK-NEXT:                   "endLine": 4,
//CHECK-NEXT:                   "startColumn": 7,
//CHECK-NEXT:                   "startLine": 4
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "message": {
//CHECK-NEXT:             "text": "zero size arrays are an extension"
//CHECK-NEXT:           },
//CHECK-NEXT:           "ruleId": "clang-diagnostic-zero-length-array",
//CHECK-NEXT:           "ruleIndex": 2
//CHECK-NEXT:         },
//CHECK-NEXT:         {
//CHECK-NEXT:           "level": "error",
//CHECK-NEXT:           "locations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 12,
//CHECK-NEXT:                   "endLine": 6,
//CHECK-NEXT:                   "startColumn": 11,
//CHECK-NEXT:                   "startLine": 6
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "message": {
//CHECK-NEXT:             "text": "unknown type name 'x'"
//CHECK-NEXT:           },
//CHECK-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-NEXT:           "ruleIndex": 1
//CHECK-NEXT:         },
//CHECK-NEXT:         {
//CHECK-NEXT:           "level": "error",
//CHECK-NEXT:           "locations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 4,
//CHECK-NEXT:                   "endLine": 8,
//CHECK-NEXT:                   "startColumn": 3,
//CHECK-NEXT:                   "startLine": 8
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "message": {
//CHECK-NEXT:             "text": "a type specifier is required for all declarations"
//CHECK-NEXT:           },
//CHECK-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-NEXT:           "ruleIndex": 1
//CHECK-NEXT:         },
//CHECK-NEXT:         {
//CHECK-NEXT:           "level": "error",
//CHECK-NEXT:           "locations": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "physicalLocation": {
//CHECK-NEXT:                 "artifactLocation": {
//CHECK-NEXT:                   "index": 0,
//CHECK-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-NEXT:                 },
//CHECK-NEXT:                 "region": {
//CHECK-NEXT:                   "endColumn": 4,
//CHECK-NEXT:                   "endLine": 9,
//CHECK-NEXT:                   "startColumn": 3,
//CHECK-NEXT:                   "startLine": 9
//CHECK-NEXT:                 }
//CHECK-NEXT:               }
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "message": {
//CHECK-NEXT:             "text": "single-argument constructors must be marked explicit to avoid unintentional implicit conversions"
//CHECK-NEXT:           },
//CHECK-NEXT:           "ruleId": "google-explicit-constructor",
//CHECK-NEXT:           "ruleIndex": 3
//CHECK-NEXT:         }
//CHECK-NEXT:       ],
//CHECK-NEXT:       "tool": {
//CHECK-NEXT:         "driver": {
//CHECK-NEXT:           "fullName": "clang-tidy",
//CHECK-NEXT:           "informationUri": "https://clang.llvm.org/extra/clang-tidy/",
//CHECK-NEXT:           "language": "en-US",
//CHECK-NEXT:           "name": "clang-tidy",
//CHECK-NEXT:           "rules": [
//CHECK-NEXT:             {
//CHECK-NEXT:               "defaultConfiguration": {
//CHECK-NEXT:                 "enabled": true,
//CHECK-NEXT:                 "level": "error",
//CHECK-NEXT:                 "rank": 50
//CHECK-NEXT:               },
//CHECK-NEXT:               "id": "clang-diagnostic-missing-prototypes",
//CHECK-NEXT:               "name": "clang-diagnostic-missing-prototypes"
//CHECK-NEXT:             },
//CHECK-NEXT:             {
//CHECK-NEXT:               "defaultConfiguration": {
//CHECK-NEXT:                 "enabled": true,
//CHECK-NEXT:                 "level": "error",
//CHECK-NEXT:                 "rank": 50
//CHECK-NEXT:               },
//CHECK-NEXT:               "id": "clang-diagnostic-error",
//CHECK-NEXT:               "name": "clang-diagnostic-error"
//CHECK-NEXT:             },
//CHECK-NEXT:             {
//CHECK-NEXT:               "defaultConfiguration": {
//CHECK-NEXT:                 "enabled": true,
//CHECK-NEXT:                 "level": "warning",
//CHECK-NEXT:                 "rank": -1
//CHECK-NEXT:               },
//CHECK-NEXT:               "id": "clang-diagnostic-zero-length-array",
//CHECK-NEXT:               "name": "clang-diagnostic-zero-length-array"
//CHECK-NEXT:             },
//CHECK-NEXT:             {
//CHECK-NEXT:               "defaultConfiguration": {
//CHECK-NEXT:                 "enabled": true,
//CHECK-NEXT:                 "level": "error",
//CHECK-NEXT:                 "rank": 50
//CHECK-NEXT:               },
//CHECK-NEXT:               "id": "google-explicit-constructor",
//CHECK-NEXT:               "name": "google-explicit-constructor"
//CHECK-NEXT:             }
//CHECK-NEXT:           ],
//CHECK-NEXT:           "version": "{{.*}}"
//CHECK-NEXT:         }
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:   ],
//CHECK-NEXT:   "version": "{{.*}}"
//CHECK-NEXT: }