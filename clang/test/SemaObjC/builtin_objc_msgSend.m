// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// rdar://8632525
extern id objc_msgSend(id self, SEL op, ...);

// rdar://12489098
struct objc_super {
  id receiver;
  Class super_class;
};

extern __attribute__((visibility("default"))) id objc_msgSendSuper(struct objc_super *super, SEL op, ...)
    __attribute__((availability(macosx,introduced=10.0)));
    
extern __attribute__((visibility("default"))) void objc_msgSendSuper_stret(struct objc_super *super, SEL op, ...)
    __attribute__((availability(macosx,introduced=10.0)));
    
extern __attribute__((visibility("default"))) void objc_msgSend_stret(id self, SEL op, ...)
    __attribute__((availability(macosx,introduced=10.0)));

