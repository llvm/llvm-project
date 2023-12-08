// RUN: %clang_cc1 -fsyntax-only -verify %s

/**
When processing @synthesize, treat ivars in a class extension the same as ivars in the class @interface, 
and treat ivars in a superclass extension the same as ivars in the superclass @interface.
In particular, when searching for an ivar to back an @synthesize, do look at ivars in the class's own class 
extension but ignore any ivars in superclass class extensions.
*/

@interface Super {
  	int ISA;
}
@end

@interface Super() {
  int Property;		// expected-note {{previously declared 'Property' here}}
}
@end

@interface SomeClass : Super {
        int interfaceIvar1;
        int interfaceIvar2;
}
@property int Property;
@property int Property1;
@end

@interface SomeClass () {
  int Property1;
}
@end

@implementation SomeClass 
@synthesize Property;	// expected-error {{property 'Property' attempting to use instance variable 'Property' declared in super class 'Super'}}
@synthesize Property1;	// OK
@end
