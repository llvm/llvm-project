// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -x c-header %s -o %t/output.symbols.json
// RUN: FileCheck %s --input-file %t/output.symbols.json

// This file is purposefully NOT valid UTF-8. The doc comment below contains a
// raw 0xD5 byte. Be careful when modifying.

/*! @brief The senderÕs storage. */
int foo(void);

// CHECK:      "docComment": {
// CHECK-NEXT:   "lines": [
// CHECK:          "text": "@brief The senderï¿½s storage. "
// CHECK:        ]
// CHECK-NEXT: },
