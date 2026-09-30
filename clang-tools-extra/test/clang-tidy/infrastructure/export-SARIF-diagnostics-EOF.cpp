// RUN: echo -n 'int main() { return 0; }' > %t-input.cpp
// RUN: clang-tidy %t-input.cpp -checks='-*,clang-diagnostic-newline-eof' --allow-no-checks -sarif-export=%t.sarif -- -Wnewline-eof > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s -implicit-check-not='{{warning|error|note}}:'
// RUN: FileCheck -input-file=%t.sarif -check-prefix=CHECK-SARIF %s

//CHECK-MESSAGES: -input.cpp:1:25: warning: no newline at end of file [clang-diagnostic-newline-eof]

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
//CHECK-SARIF-NEXT:           "level": "warning",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "file://{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 25,
//CHECK-SARIF-NEXT:                   "startColumn": 25,
//CHECK-SARIF-NEXT:                   "startLine": 1
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "no newline at end of file"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-diagnostic-newline-eof",
//CHECK-SARIF-NEXT:           "ruleIndex": 0
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
//CHECK-SARIF-NEXT:                 "level": "warning",
//CHECK-SARIF-NEXT:                 "rank": -1
//CHECK-SARIF-NEXT:               },
//CHECK-SARIF-NEXT:               "fullDescription": {
//CHECK-SARIF-NEXT:                 "text": ""
//CHECK-SARIF-NEXT:               },
//CHECK-SARIF-NEXT:               "id": "clang-diagnostic-newline-eof",
//CHECK-SARIF-NEXT:               "name": "clang-diagnostic-newline-eof"
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "version": "24.0.0git"
//CHECK-SARIF-NEXT:         }
//CHECK-SARIF-NEXT:       }
//CHECK-SARIF-NEXT:     }
//CHECK-SARIF-NEXT:   ],
//CHECK-SARIF-NEXT:   "version": "2.1.0"


