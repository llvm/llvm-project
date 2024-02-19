// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --emit-extension-symbol-graphs --symbol-graph-dir=%t/symbols \
// RUN:   --product-name=Module -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules-cache \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %t/input.h -verify

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

// RUN: Filecheck %s --input-file  %t/symbols/Module.symbols.json --check-prefix MOD
// MOD-NOT: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(py)Property $ c:objc(cs)ExtInterface"
// MOD-NOT: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(im)InstanceMethod $ c:objc(cs)ExtInterface"
// MOD-NOT: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(cm)ClassMethod $ c:objc(cs)ExtInterface"
// MOD-NOT: "!testLabel": "c:objc(cs)ExtInterface(py)Property"
// MOD-NOT: "!testLabel": "c:objc(cs)ExtInterface(im)InstanceMethod"
// MOD-NOT: "!testLabel": "c:objc(cs)ExtInterface(cm)ClassMethod"
// MOD-NOT: "!testLabel": "c:objc(cs)ExtInterface"
// MOD-DAG: "!testLabel": "c:objc(cs)ModInterface"

// RUN: Filecheck %s --input-file %t/symbols/ExternalModule@Module.symbols.json --check-prefix EXT
// EXT-DAG: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(py)Property $ c:objc(cs)ExtInterface"
// EXT-DAG: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(im)InstanceMethod $ c:objc(cs)ExtInterface"
// EXT-DAG: "!testRelLabel": "memberOf $ c:objc(cs)ExtInterface(cm)ClassMethod $ c:objc(cs)ExtInterface"
// EXT-DAG: "!testLabel": "c:objc(cs)ExtInterface(py)Property"
// EXT-DAG: "!testLabel": "c:objc(cs)ExtInterface(im)InstanceMethod"
// EXT-DAG: "!testLabel": "c:objc(cs)ExtInterface(cm)ClassMethod"
// EXT-NOT: "!testLabel": "c:objc(cs)ExtInterface"
// EXT-NOT: "!testLabel": "c:objc(cs)ModInterface"
