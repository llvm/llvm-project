// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -fmodules-cache-path=%t.mcp -fmodules -fimplicit-module-maps -F %S/Inputs -fobjc-arc %s -verify

@import Module;

__attribute__((objc_root_class))
@interface Test
// The diagnostic will try to find a suitable macro name to use (instead of raw __attribute__).
// While iterating through the macros it would dereference a null pointer if the macro was undefined in the same module as it was originally defined in.
@property (assign) id newFile; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} \
                               // expected-note {{explicitly declare getter '-newFile' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@end

@implementation Test
@end
