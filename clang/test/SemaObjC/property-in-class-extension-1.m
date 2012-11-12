// RUN: %clang_cc1  -fsyntax-only -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -verify -Weverything %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -verify -Weverything %s
// rdar://12103400

@class NSString;

@interface MyClass

@property (nonatomic, readonly) NSString* addingMemoryModel;

@property (nonatomic, copy, readonly) NSString* matchingMemoryModel;

@property (nonatomic, retain, readonly) NSString* addingNoNewMemoryModel;

@property (readonly) NSString* none;
@property (readonly) NSString* none1;

@property (assign, readonly) NSString* changeMemoryModel; // expected-note {{property declared here}}

@property (readonly) __weak id weak_prop;
@property (readonly) __weak id weak_prop1;

@property (assign, readonly) NSString* assignProperty;

@property (readonly) NSString* readonlyProp;



@end

@interface MyClass ()
{
  NSString* _name;
}

@property (nonatomic, copy) NSString* addingMemoryModel;
@property (nonatomic, copy) NSString* matchingMemoryModel;
@property () NSString* addingNoNewMemoryModel;
@property () NSString* none;
@property (readwrite, retain) NSString* none1;

@property (retain) NSString* changeMemoryModel; // expected-warning {{property attribute in class extension does not match the primary class}}
@property () __weak id weak_prop;
@property (readwrite) __weak id weak_prop1;

@property (assign, readwrite) NSString* assignProperty;
@property (assign) NSString* readonlyProp;
@end

// rdar://12214070
@interface radar12214070
@property (nonatomic, atomic, readonly) float propertyName; // expected-error {{property attributes 'atomic' and 'nonatomic' are mutually exclusive}}
@end

@interface radar12214070 ()
@property (atomic, nonatomic, readonly, readwrite) float propertyName; // expected-error {{property attributes 'readonly' and 'readwrite' are mutually exclusive}} \
		// expected-error {{property attributes 'atomic' and 'nonatomic' are mutually exclusive}}
@end

