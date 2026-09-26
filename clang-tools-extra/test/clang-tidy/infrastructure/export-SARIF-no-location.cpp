// RUN: grep -Ev "// *[A-Z-]+:" %s > %t-input.cpp
// RUN: not clang-tidy %t-input.cpp -checks='-*,clang-diagnostic-*,google-explicit-constructor' -sarif-export=%t.sarif -- -fake-command > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s -implicit-check-not='{{warning|error|note}}:'
// RUN: FileCheck -input-file=%t.sarif -check-prefix=CHECK-SARIF %s
class A { A(int) {} };

//NOTE: "-fake-command" is rejected by the driver before a SourceManager is created,
//NOTE: so the diagnostic is stored with an empty FilePath and ranges. Therefore, getResultRanges()
//NOTE: omits locations entirely. 

//CHECK-MESSAGES: error: unknown argument: '-fake-command' [clang-diagnostic-error]
//CHECK-MESSAGES: -input.cpp:1:11: warning: single-argument constructors must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]

//CHECK-SARIF: {
//CHECK-SARIF-NEXT:   "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json",
//CHECK-SARIF-NEXT:   "runs": [
//CHECK-SARIF-NEXT:     {
//CHECK-SARIF-NEXT:       "artifacts": [
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "length": {{[0-9]+}},
//CHECK-SARIF-NEXT:           "location": {
//CHECK-SARIF-NEXT:             "index": 0,
//CHECK-SARIF-NEXT:             "uri": "file://{{.*}}-input.cpp"
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
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "unknown argument: '-fake-command'"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-error",
//CHECK-SARIF-NEXT:           "ruleIndex": 0
//CHECK-SARIF-NEXT:         },
//CHECK-SARIF-NEXT:         {
//CHECK-SARIF-NEXT:           "level": "warning",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 12,
//CHECK-SARIF-NEXT:                   "endLine": 1,
//CHECK-SARIF-NEXT:                   "startColumn": 11,
//CHECK-SARIF-NEXT:                   "startLine": 1
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "single-argument constructors must be marked explicit to avoid unintentional implicit conversions"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "google-explicit-constructor",
//CHECK-SARIF-NEXT:           "ruleIndex": 1
//CHECK-SARIF-NEXT:         }
//CHECK-SARIF-NEXT:       ],
//CHECK-SARIF-NEXT:       "tool": {
//CHECK-SARIF-NEXT:         "driver": {
//CHECK-SARIF-NEXT:           "fullName": "clang-tidy",
//CHECK-SARIF-NEXT:           "informationUri": "https://clang.llvm.org/extra/clang-tidy/",
//CHECK-SARIF-NEXT:           "language": "en-US",
//CHECK-SARIF-NEXT:           "name": "clang-tidy",
//CHECK-SARIF-NEXT:           "rules": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "defaultConfiguration": {
//CHECK-SARIF-NEXT:                 "enabled": true,
//CHECK-SARIF-NEXT:                 "level": "error",
//CHECK-SARIF-NEXT:                 "rank": 50
//CHECK-SARIF-NEXT:               },
//CHECK-SARIF-NEXT:               "id": "clang-diagnostic-error",
//CHECK-SARIF-NEXT:               "name": "clang-diagnostic-error"
//CHECK-SARIF-NEXT:             },
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "defaultConfiguration": {
//CHECK-SARIF-NEXT:                 "enabled": true,
//CHECK-SARIF-NEXT:                 "level": "warning",
//CHECK-SARIF-NEXT:                 "rank": -1
//CHECK-SARIF-NEXT:               },
//CHECK-SARIF-NEXT:               "id": "google-explicit-constructor",
//CHECK-SARIF-NEXT:               "name": "google-explicit-constructor"
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "version": "{{.*}}"
//CHECK-SARIF-NEXT:         }
//CHECK-SARIF-NEXT:       }
//CHECK-SARIF-NEXT:     }
//CHECK-SARIF-NEXT:   ],
//CHECK-SARIF-NEXT:   "version": "{{.*}}"
//CHECK-SARIF-NEXT: }
