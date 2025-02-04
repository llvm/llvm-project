// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %s -o %t/output.symbols.json

_Pragma("clang assume_nonnull begin")

struct Foo { int a; };
typedef struct Foo *Bar;
// RUN: FileCheck %s -input-file %t/output.symbols.json --check-prefix FUNC
void func(Bar b);
// FUNC-LABEL: "!testLabel": "c:@F@func",
// CHECK-NOT: Foo
// CHECK: "pathComponents"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix THING
#define SWIFT_NAME(_name) __attribute__((swift_name(#_name)))
extern Bar const thing SWIFT_NAME(swiftThing);
// THING-LABEL: "!testLabel": "c:@thing"
// THING-NOT: Foo
// THING: "pathComponents"

_Pragma("clang assume_nonnull end")

// expected-no-diagnostics
