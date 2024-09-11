// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-macosx \
// RUN:   -x objective-c-header %s -o %t/output.symbols.json -verify


// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix A
__attribute__((availability(macos, introduced=9.0, deprecated=12.0, obsoleted=20.0)))
@interface A
// A-LABEL: "!testLabel": "c:objc(cs)A"
// A:      "availability": [
// A-NEXT:   {
// A-NEXT:     "deprecated": {
// A-NEXT:       "major": 12,
// A-NEXT:       "minor": 0,
// A-NEXT:       "patch": 0
// A-NEXT:     }
// A-NEXT:     "domain": "macos"
// A-NEXT:     "introduced": {
// A-NEXT:       "major": 9,
// A-NEXT:       "minor": 0,
// A-NEXT:       "patch": 0
// A-NEXT:     }
// A-NEXT:     "obsoleted": {
// A-NEXT:       "major": 20,
// A-NEXT:       "minor": 0,
// A-NEXT:       "patch": 0
// A-NEXT:     }
// A-NEXT:   }
// A-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix CP
@property(class) int CP;
// CP-LABEL: "!testLabel": "c:objc(cs)A(cpy)CP"
// CP:      "availability": [
// CP-NEXT:   {
// CP-NEXT:     "deprecated": {
// CP-NEXT:       "major": 12,
// CP-NEXT:       "minor": 0,
// CP-NEXT:       "patch": 0
// CP-NEXT:     }
// CP-NEXT:     "domain": "macos"
// CP-NEXT:     "introduced": {
// CP-NEXT:       "major": 9,
// CP-NEXT:       "minor": 0,
// CP-NEXT:       "patch": 0
// CP-NEXT:     }
// CP-NEXT:     "obsoleted": {
// CP-NEXT:       "major": 20,
// CP-NEXT:       "minor": 0,
// CP-NEXT:       "patch": 0
// CP-NEXT:     }
// CP-NEXT:   }
// CP-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix IP
@property int IP;
// IP-LABEL: "!testLabel": "c:objc(cs)A(py)IP"
// IP:      "availability": [
// IP-NEXT:   {
// IP-NEXT:     "deprecated": {
// IP-NEXT:       "major": 12,
// IP-NEXT:       "minor": 0,
// IP-NEXT:       "patch": 0
// IP-NEXT:     }
// IP-NEXT:     "domain": "macos"
// IP-NEXT:     "introduced": {
// IP-NEXT:       "major": 9,
// IP-NEXT:       "minor": 0,
// IP-NEXT:       "patch": 0
// IP-NEXT:     }
// IP-NEXT:     "obsoleted": {
// IP-NEXT:       "major": 20,
// IP-NEXT:       "minor": 0,
// IP-NEXT:       "patch": 0
// IP-NEXT:     }
// IP-NEXT:   }
// IP-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix MR
@property int moreRestrictive __attribute__((availability(macos, introduced=10.0, deprecated=11.0, obsoleted=19.0)));
// MR-LABEL: "!testLabel": "c:objc(cs)A(py)moreRestrictive"
// MR:      "availability": [
// MR-NEXT:   {
// MR-NEXT:     "deprecated": {
// MR-NEXT:       "major": 11,
// MR-NEXT:       "minor": 0,
// MR-NEXT:       "patch": 0
// MR-NEXT:     }
// MR-NEXT:     "domain": "macos"
// MR-NEXT:     "introduced": {
// MR-NEXT:       "major": 10,
// MR-NEXT:       "minor": 0,
// MR-NEXT:       "patch": 0
// MR-NEXT:     }
// MR-NEXT:     "obsoleted": {
// MR-NEXT:       "major": 19,
// MR-NEXT:       "minor": 0,
// MR-NEXT:       "patch": 0
// MR-NEXT:     }
// MR-NEXT:   }
// MR-NEXT: ]

@end

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix B
__attribute__((deprecated("B is deprecated")))
@interface B
// B-LABEL: "!testLabel": "c:objc(cs)B"
// B:      "availability": [
// B-NEXT:   {
// B-NEXT:     "domain": "*"
// B-NEXT:     "isUnconditionallyDeprecated": true
// B-NEXT:   }
// B-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix BIP
@property int BIP;
// BIP-LABEL: "!testLabel": "c:objc(cs)B(py)BIP"
// BIP:      "availability": [
// BIP-NEXT:   {
// BIP-NEXT:     "domain": "*"
// BIP-NEXT:     "isUnconditionallyDeprecated": true
// BIP-NEXT:   }
// BIP-NEXT: ]
@end

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix C
__attribute__((availability(macos, unavailable)))
@interface C
// C-LABEL: "!testLabel": "c:objc(cs)C"
// C:      "availability": [
// C-NEXT:   {
// C-NEXT:     "domain": "macos"
// C-NEXT:     "isUnconditionallyUnavailable": true
// C-NEXT:   }
// C-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix CIP
@property int CIP;
// CIP-LABEL: "!testLabel": "c:objc(cs)C(py)CIP"
// CIP:      "availability": [
// CIP-NEXT:   {
// CIP-NEXT:     "domain": "macos"
// CIP-NEXT:     "isUnconditionallyUnavailable": true
// CIP-NEXT:   }
// CIP-NEXT: ]
@end

@interface D
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix DIP
@property int DIP __attribute__((availability(macos, introduced=10.0, deprecated=11.0, obsoleted=19.0)));
// DIP-LABEL: "!testLabel": "c:objc(cs)D(py)DIP"
// DIP:      "availability": [
// DIP-NEXT:   {
// DIP-NEXT:     "deprecated": {
// DIP-NEXT:       "major": 11,
// DIP-NEXT:       "minor": 0,
// DIP-NEXT:       "patch": 0
// DIP-NEXT:     }
// DIP-NEXT:     "domain": "macos"
// DIP-NEXT:     "introduced": {
// DIP-NEXT:       "major": 10,
// DIP-NEXT:       "minor": 0,
// DIP-NEXT:       "patch": 0
// DIP-NEXT:     }
// DIP-NEXT:     "obsoleted": {
// DIP-NEXT:       "major": 19,
// DIP-NEXT:       "minor": 0,
// DIP-NEXT:       "patch": 0
// DIP-NEXT:     }
// DIP-NEXT:   }
// DIP-NEXT: ]
@end

// expected-no-diagnostics
