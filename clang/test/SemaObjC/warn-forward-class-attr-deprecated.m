// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@class ABGroupImportFilesScope; // expected-note {{forward declaration of class here}}

@interface I1
- (id) filenames __attribute__((deprecated)); // expected-note {{'filenames' has been explicitly marked deprecated here}}
@end

@interface I2
- (id) Meth : (ABGroupImportFilesScope*) scope;
- (id) filenames __attribute__((deprecated));
- (id)initWithAccount: (id)account filenames:(id)filenames;
@end

@implementation I2
- (id) Meth : (ABGroupImportFilesScope*) scope
{
  id p =  [self initWithAccount : 0 filenames :[scope filenames]]; // expected-warning {{'filenames' may be deprecated because the receiver type is unknown}}
  return 0;
}
- (id) filenames { return 0; }
- (id)initWithAccount: (id)account filenames:(id)filenames { return 0; }
@end
