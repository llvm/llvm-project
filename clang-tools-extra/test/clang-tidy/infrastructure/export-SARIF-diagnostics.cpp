// RUN: grep -Ev "// *[A-Z-]+:" %s > %t-input.cpp
// RUN: not clang-tidy %t-input.cpp -checks='-*,google-explicit-constructor,clang-diagnostic-missing-prototypes,clang-diagnostic-zero-length-array' --warnings-as-errors='clang-diagnostic-missing-prototypes,google-explicit-constructor' -export-sarif=%t.sarif -- -Wmissing-prototypes -Wzero-length-array > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s -implicit-check-not='{{warning|error|note}}:'
// RUN: FileCheck -input-file=%t.sarif -check-prefix=CHECK-SARIF %s
#define X(n) void n ## n() {}
X(f)
int a[-1];
int b[0];

void test(x);
struct Foo {
  member;
  Foo(int) {}
};

//CHECK-MESSAGES: -input.cpp:2:1: error: no previous prototype for function 'ff' [clang-diagnostic-missing-prototypes,-warnings-as-errors]
//CHECK-MESSAGES: -input.cpp:1:19: note: expanded from macro 'X'
//CHECK-MESSAGES: {{^}}note: expanded from here{{$}}
//CHECK-MESSAGES: -input.cpp:2:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
//CHECK-MESSAGES: -input.cpp:1:14: note: expanded from macro 'X'
//CHECK-MESSAGES: -input.cpp:3:7: error: 'a' declared as an array with a negative size [clang-diagnostic-error]
//CHECK-MESSAGES: -input.cpp:4:7: warning: zero size arrays are an extension [clang-diagnostic-zero-length-array]
//CHECK-MESSAGES: -input.cpp:6:11: error: unknown type name 'x' [clang-diagnostic-error]
//CHECK-MESSAGES: -input.cpp:8:3: error: a type specifier is required for all declarations [clang-diagnostic-error]
//CHECK-MESSAGES: -input.cpp:9:3: error: single-argument constructors must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor,-warnings-as-errors]

//CHECK-SARIF: {
//CHECK-SARIF-NEXT:   "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json",
//CHECK-SARIF-NEXT:   "runs": [
//CHECK-SARIF-NEXT:     {
//CHECK-SARIF-NEXT:       "artifacts": [
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "length": {{[0-9]+}},
//CHECK-SARIF-NEXT:           "location": {
//CHECK-SARIF-NEXT:             "index": 0,
//CHECK-SARIF-NEXT:             "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "mimeType": "text/plain",
//CHECK-SARIF-NEXT:           "roles": [
//CHECK-SARIF-NEXT:             "resultFile"
//CHECK-SARIF-NEXT:           ]
//CHECK-SARIF-NEXT:         }
//CHECK-SARIF-NEXT:       ],
//CHECK-SARIF-NEXT:       "columnKind": "unicodeCodePoints",
//CHECK-SARIF-NEXT:       "results": [
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "level": "error",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 2,
//CHECK-SARIF-NEXT:                   "endLine": 2,
//CHECK-SARIF-NEXT:                   "startColumn": 1,
//CHECK-SARIF-NEXT:                   "startLine": 2
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "no previous prototype for function 'ff'"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-missing-prototypes",
//CHECK-SARIF-NEXT:           "ruleIndex": 0
//CHECK-SARIF-NEXT:         },
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "level": "error",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 9,
//CHECK-SARIF-NEXT:                   "endLine": 3,
//CHECK-SARIF-NEXT:                   "startColumn": 7,
//CHECK-SARIF-NEXT:                   "startLine": 3
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "'a' declared as an array with a negative size"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-SARIF-NEXT:           "ruleIndex": 1
//CHECK-SARIF-NEXT:         },
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "level": "warning",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 8,
//CHECK-SARIF-NEXT:                   "endLine": 4,
//CHECK-SARIF-NEXT:                   "startColumn": 7,
//CHECK-SARIF-NEXT:                   "startLine": 4
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "zero size arrays are an extension"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-zero-length-array",
//CHECK-SARIF-NEXT:           "ruleIndex": 2
//CHECK-SARIF-NEXT:         },
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "level": "error",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "{{.*}}-input.cpp" 
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 12,
//CHECK-SARIF-NEXT:                   "endLine": 6,
//CHECK-SARIF-NEXT:                   "startColumn": 11,
//CHECK-SARIF-NEXT:                   "startLine": 6
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "unknown type name 'x'"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-SARIF-NEXT:           "ruleIndex": 1
//CHECK-SARIF-NEXT:         },
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "level": "error",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 4,
//CHECK-SARIF-NEXT:                   "endLine": 8,
//CHECK-SARIF-NEXT:                   "startColumn": 3,
//CHECK-SARIF-NEXT:                   "startLine": 8
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "a type specifier is required for all declarations"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-SARIF-NEXT:           "ruleIndex": 1
//CHECK-SARIF-NEXT:         },
