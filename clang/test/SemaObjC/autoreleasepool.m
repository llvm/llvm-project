// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

void *objc_autoreleasepool_push(void);
void autoreleasepool_pop(void*);

@interface AUTORP @end

@implementation AUTORP
- (void) unregisterTask:(id) task {
  goto L;	// expected-error {{cannot jump}}

  @autoreleasepool { // expected-note {{jump bypasses auto release push of @autoreleasepool block}}
        void *tmp = objc_autoreleasepool_push();
        L:
        autoreleasepool_pop(tmp);
        @autoreleasepool {
          return;
        }
  }
}
@end

