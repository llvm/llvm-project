// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://8918702

typedef void (^b_t)(void);
void a(b_t work) { }
struct _s {
    int a;
};
struct _s *r(void);

void f(void) {
    __block struct _s *s = 0;
    a(^{
        s = (struct _s *)r();
    });
}
