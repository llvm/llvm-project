// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fblocks -verify %s

// rdar://20130079

#if __has_feature(objc_arc)
__attribute__((ns_returns_retained)) id (^invalidBlockRedecl)(); // expected-note {{previous definition is here}}
id (^invalidBlockRedecl)(); //expected-error {{redefinition of 'invalidBlockRedecl' with a different type: 'id (^__strong)()' vs 'id ((^__strong))() __attribute__((ns_returns_retained))'}}
#else
__attribute__((ns_returns_retained)) id (^invalidBlockRedecl)();
id (^invalidBlockRedecl)();
#endif

typedef __attribute__((ns_returns_retained)) id (^blockType)();

typedef __attribute__((ns_returns_retained)) int (^invalidBlockType)(); // expected-warning {{'ns_returns_retained' attribute only applies to functions that return an Objective-C object}}

__attribute__((ns_returns_retained)) int functionDecl();  // expected-warning {{'ns_returns_retained' attribute only applies to functions that return an Objective-C object}}
