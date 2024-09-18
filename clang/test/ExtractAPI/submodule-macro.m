// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --emit-extension-symbol-graphs --symbol-graph-dir=%t/symbols -isystem %t \
// RUN:   --product-name=Umbrella -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules-cache \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %t/Umbrella.h %t/Subheader.h

//--- Umbrella.h
#include "Subheader.h"
#import <stdbool.h>

//--- Subheader.h
#define FOO 1

//--- module.modulemap
module Umbrella {
    umbrella header "Umbrella.h"
    export *
    module * { export * }
}

// RUN: FileCheck %s --input-file  %t/symbols/Umbrella.symbols.json --check-prefix MOD
// MOD-NOT: bool
// MOD: "!testLabel": "c:@macro@FOO"

