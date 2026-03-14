// RUN: %clang_cc1 -fsyntax-only -verify -Wdocumentation -Wno-objc-root-class %s

@interface InvalidInterface { // expected-note {{previous definition is here}}
  int *_property;
}

@end

/*!
 */

@interface InvalidInterface // expected-error {{duplicate interface definition for class 'InvalidInterface'}}
@property int *property;

-(void) method;
@end

@implementation InvalidInterface
-(void) method { }
@end
