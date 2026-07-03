// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stddef.h>

typedef struct objc_object *id;
id objc_getClass(const char *s);

@interface Object 
- (id) initWithInt: (int) i;
@end

@protocol Func
+ (int) class_func0;
- (int) instance_func0;
@end

@interface Derived: Object // expected-note {{receiver is instance of class declared here}}
+ (int) class_func1;
+ (int) class_func2;
+ (int) class_func3;
+ (int) class_func4;
+ (int) class_func5;
+ (int) class_func6;
+ (int) class_func7;
- (int) instance_func1;
- (int) instance_func2;
- (int) instance_func3;
- (int) instance_func4;
- (int) instance_func5;
- (int) instance_func6;
- (int) instance_func7;
- (id) initWithInt: (int) i;
@end

@implementation Derived
+ (int) class_func1
{
   int i = (size_t)[self class_func0];       // expected-warning {{class method '+class_func0' not found (return type defaults to 'id'); did you mean '+class_func}}
   return i + (size_t)[super class_func0];   // expected-warning {{class method '+class_func0' not found (return type defaults to 'id')}}
}
+ (int) class_func2
{
   int i = [(id <Func>)self class_func0];
   i += [(id <Func>)super class_func0];    // expected-error {{cannot cast 'super' (it isn't an expression)}}
   i += [(Class <Func>)self class_func0];  // 
   return i + [(Class <Func>)super class_func0]; // // expected-error {{cannot cast 'super' (it isn't an expression)}}
}
+ (int) class_func3
{
   return [(Object <Func> *)super class_func0];  // expected-error {{cannot cast 'super' (it isn't an expression)}}
}
+ (int) class_func4
{
   return [(Derived <Func> *)super class_func0]; // expected-error {{cannot cast 'super' (it isn't an expression)}}
}   
+ (int) class_func5
{
   int i = (size_t)[Derived class_func0];    // expected-warning {{class method '+class_func0' not found (return type defaults to 'id')}}
   return i + (size_t)[Object class_func0];  // expected-warning {{class method '+class_func0' not found (return type defaults to 'id')}}
}
+ (int) class_func6
{
   return (size_t)[objc_getClass("Object") class_func1]; // GCC warns about this
}
+ (int) class_func7
{
   return [objc_getClass("Derived") class_func1];
}
- (int) instance_func1
{
   int i = (size_t)[self instance_func0];     // expected-warning {{instance method '-instance_func0' not found (return type defaults to 'id'); did you mean}}
   return i + (size_t)[super instance_func0]; // expected-warning {{'Object' may not respond to 'instance_func0'}}
}
- (int) instance_func2
{
   return [(id <Func>)super instance_func0]; // expected-error {{cannot cast 'super' (it isn't an expression)}}
}
- (int) instance_func3
{
   return [(Object <Func> *)super instance_func0]; // expected-error {{cannot cast 'super' (it isn't an expression)}}
}
- (int) instance_func4
{
   return [(Derived <Func> *)super instance_func0]; // expected-error {{cannot cast 'super' (it isn't an expression)}}
}   
- (int) instance_func5
{
   int i = (size_t)[Derived instance_func1]; // expected-warning {{class method '+instance_func1' not found (return type defaults to 'id')}} 
   return i + (size_t)[Object instance_func1]; // expected-warning {{class method '+instance_func1' not found (return type defaults to 'id')}}
}
- (int) instance_func6
{
   return (size_t)[objc_getClass("Object") class_func1];
}
- (int) instance_func7
{
   return [objc_getClass("Derived") class_func1];
}
- (id) initWithInt: (int) i
{
   // Don't warn about parentheses here.
   if (self = [super initWithInt: i]) {
     [self instance_func1];
   }
   return self;
}
@end

@class C;
@interface A // expected-note {{receiver is instance of class declared here}}
- (instancetype)initWithCoder:(A *)coder;
@end

@interface B : A
@end

@implementation B
- (instancetype)initWithCoder:(C *)coder {     // expected-note {{'coder' declared here}}
  if (0 != (self = [super initWithCode:code])) // expected-error {{use of undeclared identifier 'code'}} expected-warning {{instance method '-initWithCode:' not found}}
    return (void *)0;
  return (void *)0;
}
@end
