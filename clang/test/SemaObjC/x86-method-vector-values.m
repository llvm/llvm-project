// RUN: %clang_cc1 -verify -DMAC -triple=i686-apple-macosx10.10 -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -DMAC -triple=i686-apple-macosx10.4 -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -DMAC -triple=i686-apple-darwin14 -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -triple=i686-apple-ios8 -Wno-objc-root-class %s

// RUN: %clang_cc1 -verify -DALLOW -DMAC -triple=i686-apple-macosx10.11 -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -DALLOW -DMAC -triple=i686-apple-darwin15 -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -DALLOW -DIOS -triple=i686-apple-ios9 -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -DALLOW -DOTHER -triple=i686-apple-watchos -Wno-objc-root-class %s
// RUN: %clang_cc1 -verify -DALLOW -DOTHER -triple=i686-apple-tvos -Wno-objc-root-class %s

// RUN: %clang_cc1 -verify -DALLOW -DOTHER -triple=x86_64-apple-macosx10.10 -Wno-objc-root-class %s

// rdar://21662309

typedef __attribute__((__ext_vector_type__(3))) float float3;

typedef float __m128 __attribute__((__vector_size__(16)));

struct Aggregate { __m128 v; };
struct AggregateFloat { float v; };

#define AVAILABLE_MACOS_10_10 __attribute__((availability(macos, introduced = 10.10)))
#define AVAILABLE_MACOS_10_11 __attribute__((availability(macos, introduced = 10.11)))

#define AVAILABLE_IOS_8 __attribute__((availability(ios, introduced = 8.0)))
#define AVAILABLE_IOS_9 __attribute__((availability(ios, introduced = 9.0)))

@interface VectorMethods

-(void)takeVector:(float3)v; // there should be no diagnostic at declaration
-(void)takeM128:(__m128)v;

@end

@implementation VectorMethods

#ifndef ALLOW

-(void)takeVector:(float3)v {
#ifdef MAC
  // expected-error@-2 {{'float3' (vector of 3 'float' values) parameter type is unsupported; support for vector types for this target is introduced in macOS 10.11}}
#else
  // expected-error@-4 {{'float3' (vector of 3 'float' values) parameter type is unsupported; support for vector types for this target is introduced in iOS 9}}
#endif
}

-(float3)retVector { // expected-error {{'float3' (vector of 3 'float' values) return type is unsupported}}
}

-(void)takeVector2:(float3)v AVAILABLE_MACOS_10_10 { // expected-error {{'float3' (vector of 3 'float' values) parameter type is unsupported}}
}

-(void)takeVector3:(float3)v AVAILABLE_MACOS_10_11 { // expected-error {{'float3' (vector of 3 'float' values) parameter type is unsupported}}
}

-(void)takeVector4:(float3)v AVAILABLE_IOS_8 { // expected-error {{'float3' (vector of 3 'float' values) parameter type is unsupported}}
}

-(void)takeVector5:(float3)v AVAILABLE_IOS_9 { // expected-error {{'float3' (vector of 3 'float' values) parameter type is unsupported}}
}

- (__m128)retM128 { // expected-error {{'__m128' (vector of 4 'float' values) return type is unsupported}}
}

- (void)takeM128:(__m128)v { // expected-error {{'__m128' (vector of 4 'float' values) parameter type is unsupported}}
}

#else

-(void)takeVector:(float3)v {
}

-(float3)retVector {
  return 0;
}

- (__m128)retM128 {
  __m128 value;
  return value;
}

- (void)takeM128:(__m128)v {
}

-(void)takeVector2:(float3)v AVAILABLE_MACOS_10_10 {
#ifdef MAC
// expected-error@-2 {{'float3' (vector of 3 'float' values) parameter type is unsupported}}
#endif
}

- (__m128)retM128_2 AVAILABLE_MACOS_10_10 {
#ifdef MAC
// expected-error@-2 {{'__m128' (vector of 4 'float' values) return type is unsupported}}
#endif
  __m128 value;
  return value;
}

-(void)takeVector3:(float3)v AVAILABLE_MACOS_10_11 { // no error
}

-(void)takeVector4:(float3)v AVAILABLE_IOS_8 {
#ifdef IOS
  // expected-error@-2 {{'float3' (vector of 3 'float' values) parameter type is unsupported}}
#endif
}

-(void)takeVector5:(float3)v AVAILABLE_IOS_9 { // no error
}

#ifdef OTHER
// expected-no-diagnostics
#endif

#endif

-(void)doStuff:(int)m { // no error
}

-(struct Aggregate)takesAndRetVectorInAggregate:(struct Aggregate)f { // no error
  struct Aggregate result;
  return result;
}

-(struct AggregateFloat)takesAndRetFloatInAggregate:(struct AggregateFloat)f { // no error
  struct AggregateFloat result;
  return result;
}


@end
