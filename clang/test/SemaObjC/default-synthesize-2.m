// RUN: %clang_cc1 -x objective-c -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://8843851

@interface StopAccessingIvarsDirectlyExample
@property(strong) id name, rank, serialNumber;
@end

@implementation StopAccessingIvarsDirectlyExample

- (void)identifyYourSelf {
    if (self.name && self.rank && self.serialNumber)
      self.name = 0;
}

// @synthesize name, rank, serialNumber;
// default synthesis allows direct access to property ivars.
- (id)init {
        _name = _rank = _serialNumber = 0;
	return self;
}

- (void)dealloc {	
}
@end


// Test2
@interface Test2 
@property(strong, nonatomic) id object;
@end

// object has user declared setter/getter so it won't be
// default synthesized; thus causing user error.
@implementation Test2
- (id) bar { return object; } // expected-error {{use of undeclared identifier 'object'}}
- (void)setObject:(id)newObject {}
- (id)object { return 0; }
@end

// Test3
@interface Test3 
{ 
  id uid;  // expected-note {{instance variable is declared here}}
} 
@property (readwrite, assign) id uid;  // expected-note {{property declared here}}
@end

// rdar://11671080
@implementation Test3 // expected-warning {{autosynthesized property 'uid' will use synthesized instance variable '_uid', not existing instance variable 'uid'}}
// Oops, forgot to write @synthesize! will be default synthesized
- (void) myMethod { 
   self.uid = 0; // Use of the “setter” 
   uid = 0; // Use of the wrong instance variable
   _uid = 0; // Use of the property instance variable
} 
@end

@interface Test4 { 
  id _var;
} 
@property (readwrite, assign) id var; 
@end


// default synthesize property named 'var'
@implementation Test4 
- (id) myMethod {
  return self->_var;  //  compiles because 'var' is synthesized by default
}
@end

@interface Test5 
{ 
  id _var;
} 
@property (readwrite, assign) id var; 
@end

// default synthesis of property 'var'
@implementation Test5 
- (id) myMethod {
  Test5 *foo = 0; 
  return foo->_var; // OK
} 
@end

@interface Test6 
{ 
  id _var; // expected-note {{'_var' declared here}}
} 
@property (readwrite, assign) id var; 
@end

// no default synthesis. So error is expected.
@implementation Test6 
- (id) myMethod 
{
  return var; // expected-error {{use of undeclared identifier 'var'}}
} 
@synthesize var = _var; 
@end

int* _object;

@interface Test7
@property (readwrite, assign) id object; 
@end

// With default synthesis, '_object' is be the synthesized ivar not the global
// 'int*' object. So no error.
@implementation Test7 
- (id) myMethod {
  return _object;
} 
@end

// rdar://11671080
@interface Test8
{
  id _y;
  id y; // expected-note {{instance variable is declared here}}
}
@property(copy) id y; // expected-note {{property declared here}}
@end


@implementation Test8 @end // expected-warning {{autosynthesized property 'y' will use  instance variable '_y', not existing instance variable 'y'}}

