// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://9078584

@interface NSObject @end

@protocol TextInput
-editRange;
@end

@interface I {
  NSObject<TextInput>* editor;
}
- (id) Meth;
@end

@implementation I
- (id) Meth {
   return editor.editRange;
}
@end

