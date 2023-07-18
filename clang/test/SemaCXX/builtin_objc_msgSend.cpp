// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// rdar://8686888

typedef struct objc_selector *SEL;
typedef struct objc_object *id;

extern "C" __attribute__((visibility("default"))) id objc_msgSend(id self, SEL op, ...)
    __attribute__((visibility("default")));

inline void TCFReleaseGC(void * object)
{
 static SEL SEL_release;
 objc_msgSend((id)object, SEL_release);
}
