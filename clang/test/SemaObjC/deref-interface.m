// RUN: %clang_cc1 -verify -fsyntax-only -Wno-objc-root-class %s

@interface NSView 
  - (id)initWithView:(id)realView;
@end

@implementation NSView
 - (id)initWithView:(id)realView {
     *(NSView *)self = *(NSView *)realView;	// expected-error {{cannot assign to class object}}
 }
@end

