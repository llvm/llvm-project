// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// rdar://8632525
extern id objc_msgSend(id self, SEL op, ...);
