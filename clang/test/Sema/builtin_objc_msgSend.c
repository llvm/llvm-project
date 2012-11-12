// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// rdar://8632525

typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;


typedef struct objc_selector *SEL;
extern id objc_msgSend(id self, SEL op, ...);

