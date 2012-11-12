// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface IDELogNavigator
{
  id selectedObjects;
}
@end

@interface IDELogNavigator (CAT)
  @property (readwrite, retain) id selectedObjects; // expected-note {{property declared here}}
  @property (readwrite, retain) id d_selectedObjects; // expected-note {{property declared here}}
@end

@implementation IDELogNavigator 
@synthesize selectedObjects = _selectedObjects; // expected-error {{property declared in category 'CAT' cannot be implemented in class implementation}}
@dynamic d_selectedObjects; // expected-error {{property declared in category 'CAT' cannot be implemented in class implementation}}
@end

