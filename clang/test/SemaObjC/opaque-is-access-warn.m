// RUN: %clang -target x86_64-apple-darwin -arch arm64 -mios-version-min=7 -fsyntax-only -Wdeprecated-objc-isa-usage %s -Xclang -verify
// RUN: %clang -target x86_64-apple-darwin -arch arm64 -mios-version-min=7 -fsyntax-only %s -Xclang -verify
// RUN: %clang -target x86_64-apple-darwin -mios-simulator-version-min=7 -fsyntax-only -Wdeprecated-objc-isa-usage %s -Xclang -verify
// RUN: %clang -target x86_64-apple-darwin -arch armv7k -mwatchos-version-min=2 -fsyntax-only -Wdeprecated-objc-isa-usage %s -Xclang -verify
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -fsyntax-only -Wdeprecated-objc-isa-usage %s -Xclang -verify

typedef struct objc_object {
  struct objc_class *isa;
} *id;

@interface NSObject {
  struct objc_class *isa;
}
@end
@interface Whatever : NSObject
+self;
@end

static void func(void) {

  id x;

  [(*x).isa self]; // expected-error {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
  [x->isa self];   // expected-error {{direct access to Objective-C's isa is deprecated in favor of object_getClass()}}
}
