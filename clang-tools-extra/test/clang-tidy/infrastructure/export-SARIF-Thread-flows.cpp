// RUN: grep -Ev "// *[A-Z-]+:" %s > %t-input.cpp
// RUN: clang-tidy %t-input.cpp -checks='-*,clang-analyzer*' -export-sarif=%t.sarif > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s -implicit-check-not='{{warning|error|note}}:'
// RUN: FileCheck -input-file=%t.sarif -check-prefix=CHECK-SARIF %s
void f() {
  int *ptr = nullptr;
  *ptr = 1;
}

//CHECK-MESSAGES: -input.cpp:3:8: warning: Dereference of null pointer (loaded from variable 'ptr') [clang-analyzer-core.NullDereference]
//CHECK-MESSAGES: -input.cpp:2:3: note: 'ptr' initialized to a null pointer value
//CHECK-MESSAGES: -input.cpp:3:8: note: Dereference of null pointer (loaded from variable 'ptr')

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
//CHECK-SARIF-NEXT:           "codeFlows": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "threadFlows": [
//CHECK-SARIF-NEXT:                 {
//CHECK-SARIF-NEXT:                   "locations": [
//CHECK-SARIF-NEXT:                     {
//CHECK-SARIF-NEXT:                       "importance": "important",
//CHECK-SARIF-NEXT:                       "location": {
//CHECK-SARIF-NEXT:                         "message": {
//CHECK-SARIF-NEXT:                           "text": "'ptr' initialized to a null pointer value"
//CHECK-SARIF-NEXT:                         },
//CHECK-SARIF-NEXT:                         "physicalLocation": {
//CHECK-SARIF-NEXT:                           "artifactLocation": {
//CHECK-SARIF-NEXT:                             "index": 0,
//CHECK-SARIF-NEXT:                             "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                           },
//CHECK-SARIF-NEXT:                           "region": {
//CHECK-SARIF-NEXT:                             "endColumn": 11,
//CHECK-SARIF-NEXT:                             "endLine": 2,
//CHECK-SARIF-NEXT:                             "startColumn": 3,
//CHECK-SARIF-NEXT:                             "startLine": 2
//CHECK-SARIF-NEXT:                           }
//CHECK-SARIF-NEXT:                         }
//CHECK-SARIF-NEXT:                       }
//CHECK-SARIF-NEXT:                     },
//CHECK-SARIF-NEXT:                     {
//CHECK-SARIF-NEXT:                       "importance": "important",
//CHECK-SARIF-NEXT:                       "location": {
//CHECK-SARIF-NEXT:                         "message": {
//CHECK-SARIF-NEXT:                           "text": "Dereference of null pointer (loaded from variable 'ptr')"
//CHECK-SARIF-NEXT:                         },
//CHECK-SARIF-NEXT:                         "physicalLocation": {
//CHECK-SARIF-NEXT:                           "artifactLocation": {
//CHECK-SARIF-NEXT:                             "index": 0,
//CHECK-SARIF-NEXT:                              "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                           },
//CHECK-SARIF-NEXT:                           "region": {
//CHECK-SARIF-NEXT:                             "endColumn": 7,
//CHECK-SARIF-NEXT:                             "endLine": 3,
//CHECK-SARIF-NEXT:                             "startColumn": 4,
//CHECK-SARIF-NEXT:                             "startLine": 3
//CHECK-SARIF-NEXT:                           }
//CHECK-SARIF-NEXT:                         }
//CHECK-SARIF-NEXT:                       }
//CHECK-SARIF-NEXT:                     }                      
//CHECK-SARIF-NEXT:                   ]        
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               ]
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "level": "warning",
//CHECK-SARIF-NEXT:           "locations": [
//CHECK-SARIF-NEXT:             {
//CHECK-SARIF-NEXT:               "physicalLocation": {
//CHECK-SARIF-NEXT:                 "artifactLocation": {
//CHECK-SARIF-NEXT:                   "index": 0,
//CHECK-SARIF-NEXT:                   "uri": "{{.*}}-input.cpp"
//CHECK-SARIF-NEXT:                 },
//CHECK-SARIF-NEXT:                 "region": {
//CHECK-SARIF-NEXT:                   "endColumn": 7,
//CHECK-SARIF-NEXT:                   "endLine": 3,
//CHECK-SARIF-NEXT:                   "startColumn": 4,
//CHECK-SARIF-NEXT:                   "startLine": 3
//CHECK-SARIF-NEXT:                 }
//CHECK-SARIF-NEXT:               }
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "message": {
//CHECK-SARIF-NEXT:             "text": "Dereference of null pointer (loaded from variable 'ptr')"
//CHECK-SARIF-NEXT:           },
//CHECK-SARIF-NEXT:           "ruleId": "clang-analyzer-core.NullDereference",
//CHECK-SARIF-NEXT:           "ruleIndex": 0
//CHECK-SARIF-NEXT:         }
//CHECK-SARIF-NEXT:       ],
//CHECK-SARIF-NEXT:       "tool": {
//CHECK-SARIF-NEXT:         "driver": {
//CHECK-SARIF-NEXT:           "fullName": "clang-tidy",
//CHECK-SARIF-NEXT:           "informationUri": "https://clang.llvm.org/docs/UsersManual.html",
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
//CHECK-SARIF-NEXT:                 "text": "Dereference of null pointer (loaded from variable 'ptr')"
//CHECK-SARIF-NEXT:               },
//CHECK-SARIF-NEXT:               "helpUri": "https://clang.llvm.org/extra/clang-tidy/checks/clang-analyzer/core.NullDereference.html",
//CHECK-SARIF-NEXT:               "id": "clang-analyzer-core.NullDereference",
//CHECK-SARIF-NEXT:               "name": "clang-analyzer-core.NullDereference"
//CHECK-SARIF-NEXT:             }
//CHECK-SARIF-NEXT:           ],
//CHECK-SARIF-NEXT:           "version": "{{.*}}"
//CHECK-SARIF-NEXT:         }
//CHECK-SARIF-NEXT:       }
//CHECK-SARIF-NEXT:     }
//CHECK-SARIF-NEXT:   ],
//CHECK-SARIF-NEXT:   "version": "{{.*}}"
//CHECK-SARIF-NEXT: }

