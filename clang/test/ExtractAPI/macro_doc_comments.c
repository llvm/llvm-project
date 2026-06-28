// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing --product-name=MacroDoc -triple arm64-apple-macosx \
// RUN:   -fretain-comments-from-system-headers -isystem %S -x objective-c-header %s -o %t/output.symbols.json

// Verify documentation comments attached to `#define` macros end up in the
// symbol graph as a `docComment` block.

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix HELLO
/// The greeting count.
#define HELLO 1
// HELLO-LABEL: "!testLabel": "c:@macro@HELLO"
// HELLO:      "docComment": {
// HELLO-NEXT:   "lines": [
// HELLO:          "text": "The greeting count."
// HELLO:        ]
// HELLO-NEXT: },

// A C-style block doc comment also attaches.
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix BLOCK
/**
 * \brief A version number.
 */
#define VERSION 42
// BLOCK-LABEL: "!testLabel": "c:@macro@VERSION"
// BLOCK:      "docComment": {
// BLOCK-NEXT:   "lines": [
// BLOCK:          "text": " \\brief A version number."
// BLOCK:        ]
// BLOCK-NEXT: },

// Function-like macros pick up their preceding doc comment.
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix SUM
/// Add two numbers.
#define SUM(x, y) ((x) + (y))
// SUM-LABEL: "!testLabel": "c:@macro@SUM"
// SUM:      "docComment": {
// SUM-NEXT:   "lines": [
// SUM:          "text": "Add two numbers."
// SUM:        ]
// SUM-NEXT: },

// Trailing `///<` doc comments attach to macros, just like fields/enumerators.
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix TRAIL
#define ANSWER 42 ///< The answer.
// TRAIL-LABEL: "!testLabel": "c:@macro@ANSWER"
// TRAIL:      "docComment": {
// TRAIL-NEXT:   "lines": [
// TRAIL:          "text": "The answer."
// TRAIL:        ]
// TRAIL-NEXT: },

// A macro with no documentation comment must NOT have a `docComment` key.
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix BARE
#define BARE 7
// BARE-LABEL: "!testLabel": "c:@macro@BARE"
// BARE-NOT:   "docComment"
// BARE:       "kind":

// A non-doc `//` comment must not be attached as a docComment.
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix PLAIN
// just an ordinary comment, not Doxygen
#define PLAIN 8
// PLAIN-LABEL: "!testLabel": "c:@macro@PLAIN"
// PLAIN-NOT:   "docComment"
// PLAIN:       "kind":

// expected-no-diagnostics
