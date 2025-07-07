// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --emit-extension-symbol-graphs --symbol-graph-dir=%t/symbols \
// RUN:   --product-name=Module -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules-cache \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %t/input.h -verify
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --product-name=Module -o %t/ModuleNoExt.symbols.json -triple arm64-apple-macosx \
// RUN:   -x objective-c-header %t/input.h

//--- input.h
#include "ExternalModule.h"

@interface ExtInterface (Category)
@property int Property;
- (void)InstanceMethod;
+ (void)ClassMethod;
@end

@interface ModInterface
@end

// expected-no-diagnostics

//--- ExternalModule.h
@interface ExtInterface
@end

//--- module.modulemap
module ExternalModule {
    header "ExternalModule.h"
}

// Main symbol graph from the build with extension SGFs
// RUN: FileCheck %s --input-file  %t/symbols/Module.symbols.json --check-prefix MOD

// MOD-NOT: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(py)Property $ c:objc(cs)ExtInterface"
// MOD-NOT: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(im)InstanceMethod $ c:objc(cs)ExtInterface"
// MOD-NOT: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(cm)ClassMethod $ c:objc(cs)ExtInterface"
// MOD-NOT: "c:objc(cs)ExtInterface(py)Property"
// MOD-NOT: "c:objc(cs)ExtInterface(im)InstanceMethod"
// MOD-NOT: "c:objc(cs)ExtInterface(cm)ClassMethod"
// MOD-NOT: "c:objc(cs)ExtInterface"
// MOD-DAG: "c:objc(cs)ModInterface"

// Symbol graph from the build without extension SGFs should be identical to main symbol graph with extension SGFs
// RUN: diff %t/symbols/Module.symbols.json %t/ModuleNoExt.symbols.json

// RUN: FileCheck %s --input-file %t/symbols/Module@ExternalModule.symbols.json --check-prefix EXT
// EXT-DAG: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(py)Property $ c:objc(cs)ExtInterface"
// EXT-DAG: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(im)InstanceMethod $ c:objc(cs)ExtInterface"
// EXT-DAG: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(cm)ClassMethod $ c:objc(cs)ExtInterface"
// EXT-DAG: "!testLabel": "c:objc(cs)ExtInterface(py)Property"
// EXT-DAG: "!testLabel": "c:objc(cs)ExtInterface(im)InstanceMethod"
// EXT-DAG: "!testLabel": "c:objc(cs)ExtInterface(cm)ClassMethod"
// EXT-NOT: "!testLabel": "c:objc(cs)ExtInterface"
// EXT-NOT: "!testLabel": "c:objc(cs)ModInterface"

// Ensure that the 'module' metadata for the extension symbol graph should still reference the
// declaring module

// RUN: FileCheck %s --input-file %t/symbols/Module@ExternalModule.symbols.json --check-prefix META
// META:       "module": {
// META-NEXT:    "name": "Module",
