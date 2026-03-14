// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// pr5986

@interface Test {
  int index;
}
- (int) index;
+ (int) ClassMethod;
@end

@implementation Test
- (int) index
{
  return index;
}
+ (int) ClassMethod
{
  return index;	// expected-error {{instance variable 'index' accessed in class method}}
}
@end

@interface Test1 {
}
- (int) InstMethod;
+ (int) ClassMethod;
@end

@implementation Test1
- (int) InstMethod
{
  return index;	// expected-error {{call to undeclared library function 'index'}}	\
                // expected-note {{include the header <strings.h> or explicitly provide a declaration for 'index'}} \
                // expected-error {{incompatible pointer to integer conversion returning}}
}
+ (int) ClassMethod
{
  return index; // expected-error {{incompatible pointer to integer conversion returning}}
}
@end

