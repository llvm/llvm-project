// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -x objective-c-header -triple arm64-apple-macosx %s -o %t/output.symbols.json -verify

@class NSString;
@protocol NSCopying
@end

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix RET_NULLABLE
NSString *_Nullable returnsNullable(void);
// RET_NULLABLE-LABEL: "!testLabel": "c:@F@returnsNullable"
// RET_NULLABLE:      "declarationFragments": [
// RET_NULLABLE-NEXT:   {
// RET_NULLABLE-NEXT:     "kind": "typeIdentifier",
// RET_NULLABLE-NEXT:     "preciseIdentifier": "c:objc(cs)NSString",
// RET_NULLABLE-NEXT:     "spelling": "NSString"
// RET_NULLABLE-NEXT:   },
// RET_NULLABLE-NEXT:   {
// RET_NULLABLE-NEXT:     "kind": "text",
// RET_NULLABLE-NEXT:     "spelling": " * "
// RET_NULLABLE-NEXT:   },
// RET_NULLABLE-NEXT:   {
// RET_NULLABLE-NEXT:     "kind": "keyword",
// RET_NULLABLE-NEXT:     "spelling": "_Nullable"
// RET_NULLABLE-NEXT:   },
// RET_NULLABLE-NEXT:   {
// RET_NULLABLE-NEXT:     "kind": "text",
// RET_NULLABLE-NEXT:     "spelling": " "
// RET_NULLABLE-NEXT:   },
// RET_NULLABLE-NEXT:   {
// RET_NULLABLE-NEXT:     "kind": "identifier",
// RET_NULLABLE-NEXT:     "spelling": "returnsNullable"
// RET_NULLABLE-NEXT:   },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix PARAM_NONNULL
void takesNonnull(NSString *_Nonnull s);
// PARAM_NONNULL-LABEL: "!testLabel": "c:@F@takesNonnull"
// PARAM_NONNULL:      "declarationFragments": [
// PARAM_NONNULL:   {
// PARAM_NONNULL:     "kind": "typeIdentifier",
// PARAM_NONNULL:     "preciseIdentifier": "c:objc(cs)NSString",
// PARAM_NONNULL:     "spelling": "NSString"
// PARAM_NONNULL-NEXT:   },
// PARAM_NONNULL-NEXT:   {
// PARAM_NONNULL-NEXT:     "kind": "text",
// PARAM_NONNULL-NEXT:     "spelling": " * "
// PARAM_NONNULL-NEXT:   },
// PARAM_NONNULL-NEXT:   {
// PARAM_NONNULL-NEXT:     "kind": "keyword",
// PARAM_NONNULL-NEXT:     "spelling": "_Nonnull"
// PARAM_NONNULL-NEXT:   },
// PARAM_NONNULL-NEXT:   {
// PARAM_NONNULL-NEXT:     "kind": "text",
// PARAM_NONNULL-NEXT:     "spelling": " "
// PARAM_NONNULL-NEXT:   },
// PARAM_NONNULL-NEXT:   {
// PARAM_NONNULL-NEXT:     "kind": "internalParam",
// PARAM_NONNULL-NEXT:     "spelling": "s"
// PARAM_NONNULL-NEXT:   },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix PLAIN_C
int *_Nullable plainCPointer(int *_Nonnull p);
// PLAIN_C-LABEL: "!testLabel": "c:@F@plainCPointer"
// PLAIN_C:      "declarationFragments": [
// PLAIN_C:   {
// PLAIN_C:     "kind": "typeIdentifier",
// PLAIN_C:     "spelling": "int"
// PLAIN_C-NEXT:   },
// PLAIN_C-NEXT:   {
// PLAIN_C-NEXT:     "kind": "text",
// PLAIN_C-NEXT:     "spelling": " * "
// PLAIN_C-NEXT:   },
// PLAIN_C-NEXT:   {
// PLAIN_C-NEXT:     "kind": "keyword",
// PLAIN_C-NEXT:     "spelling": "_Nullable"
// PLAIN_C-NEXT:   },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix NULL_UNSPEC
void nullUnspec(NSString *_Null_unspecified s);
// NULL_UNSPEC-LABEL: "!testLabel": "c:@F@nullUnspec"
// NULL_UNSPEC:      "declarationFragments": [
// NULL_UNSPEC:   {
// NULL_UNSPEC:     "kind": "keyword",
// NULL_UNSPEC:     "spelling": "_Null_unspecified"
// NULL_UNSPEC-NEXT:   },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix ID_NULLABLE
void takesIdNullable(id _Nullable obj);
// ID_NULLABLE-LABEL: "!testLabel": "c:@F@takesIdNullable"
// ID_NULLABLE:      "declarationFragments": [
// ID_NULLABLE:   {
// ID_NULLABLE:     "kind": "keyword",
// ID_NULLABLE:     "spelling": "id"
// ID_NULLABLE-NEXT:   },
// ID_NULLABLE-NEXT:   {
// ID_NULLABLE-NEXT:     "kind": "text",
// ID_NULLABLE-NEXT:     "spelling": " "
// ID_NULLABLE-NEXT:   },
// ID_NULLABLE-NEXT:   {
// ID_NULLABLE-NEXT:     "kind": "keyword",
// ID_NULLABLE-NEXT:     "spelling": "_Nullable"
// ID_NULLABLE-NEXT:   },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix ID_PROTO
void takesIdProto(id<NSCopying> _Nullable obj);
// ID_PROTO-LABEL: "!testLabel": "c:@F@takesIdProto"
// ID_PROTO:      "declarationFragments": [
// ID_PROTO:   {
// ID_PROTO:     "kind": "typeIdentifier",
// ID_PROTO:     "preciseIdentifier": "c:Qoobjc(pl)NSCopying",
// ID_PROTO:     "spelling": "id<NSCopying>"
// ID_PROTO-NEXT:   },
// ID_PROTO-NEXT:   {
// ID_PROTO-NEXT:     "kind": "text",
// ID_PROTO-NEXT:     "spelling": " "
// ID_PROTO-NEXT:   },
// ID_PROTO-NEXT:   {
// ID_PROTO-NEXT:     "kind": "keyword",
// ID_PROTO-NEXT:     "spelling": "_Nullable"
// ID_PROTO-NEXT:   },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix ASSUME_NN
_Pragma("clang assume_nonnull begin")
NSString *implicitlyNonnull(NSString *s);
_Pragma("clang assume_nonnull end")
// ASSUME_NN-LABEL: "!testLabel": "c:@F@implicitlyNonnull"
// ASSUME_NN:      "declarationFragments": [
// ASSUME_NN:   {
// ASSUME_NN:     "kind": "typeIdentifier",
// ASSUME_NN:     "preciseIdentifier": "c:objc(cs)NSString",
// ASSUME_NN:     "spelling": "NSString"
// ASSUME_NN-NEXT:   },
// ASSUME_NN-NEXT:   {
// ASSUME_NN-NEXT:     "kind": "text",
// ASSUME_NN-NEXT:     "spelling": " * "
// ASSUME_NN-NEXT:   },
// ASSUME_NN-NEXT:   {
// ASSUME_NN-NEXT:     "kind": "keyword",
// ASSUME_NN-NEXT:     "spelling": "_Nonnull"
// ASSUME_NN-NEXT:   },

// expected-no-diagnostics
