// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fblocks -verify %s

// rdar://20130079

#if __has_feature(objc_arc)
__attribute__((ns_returns_retained)) id (^invalidBlockRedecl)(void); // expected-note {{previous definition is here}}
id (^invalidBlockRedecl)(void); //expected-error {{redefinition of 'invalidBlockRedecl' with a different type: 'id (^__strong)(void)' vs 'id ((^__strong))(void) __attribute__((ns_returns_retained))'}}
#else
__attribute__((ns_returns_retained)) id (^invalidBlockRedecl)(void);
id (^invalidBlockRedecl)(void);
#endif

typedef __attribute__((ns_returns_retained)) id (^blockType)(void);

typedef __attribute__((ns_returns_retained)) int (^invalidBlockType)(void); // expected-warning {{'ns_returns_retained' attribute only applies to functions that return an Objective-C object}}

__attribute__((ns_returns_retained)) int functionDecl(void);  // expected-warning {{'ns_returns_retained' attribute only applies to functions that return an Objective-C object}}
