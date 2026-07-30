// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

// objc_super has special lookup rules for compatibility with macOS headers, so
// the following should compile.
struct objc_super {};
extern "C" id objc_msgSendSuper(struct objc_super *super, SEL op, ...);
extern "C" void objc_msgSendSuper_stret(struct objc_super *super, SEL op, ...);
