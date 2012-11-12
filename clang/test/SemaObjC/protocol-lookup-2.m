// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
@interface NSObject @end

@protocol ProtocolA

+ (id)classMethod;
- (id)instanceMethod;

@end

@protocol ProtocolB <ProtocolA>

@end

@interface Foo : NSObject <ProtocolB>

@end

@interface SubFoo : Foo

@end

@implementation SubFoo

+ (id)method {
  return [super classMethod];
}

- (id)method {
  return [super instanceMethod];
}

@end
