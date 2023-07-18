// RUN: %clang_cc1 -Wmethod-signatures -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -Wmethod-signatures -fsyntax-only -verify -Wno-objc-root-class %s

@interface A @end
@interface B : A @end

@interface Test1 {}
- (void) test1:(A*) object; // expected-note {{previous definition is here}} 
- (void) test2:(B*) object;
@end

@implementation Test1
- (void) test1:(B*) object {} // expected-warning {{conflicting parameter types in implementation of 'test1:': 'A *' vs 'B *'}}
- (void) test2:(A*) object {}
@end

@interface Test2 {}
- (void) test1:(id) object; // expected-note {{previous definition is here}} 
- (void) test2:(A*) object;
@end

@implementation Test2
- (void) test1:(A*) object {} // expected-warning {{conflicting parameter types in implementation of 'test1:': 'id' vs 'A *'}}
- (void) test2:(id) object {}
@end

@interface Test3 {}
- (A*) test1;
- (B*) test2; // expected-note {{previous definition is here}} 
@end

@implementation Test3
- (B*) test1 { return 0; }
- (A*) test2 { return 0; } // expected-warning {{conflicting return type in implementation of 'test2': 'B *' vs 'A *'}}
@end

// The particular case of overriding with an id return is permitted.
@interface Test4 {}
- (id) test1;
- (A*) test2;
@end
@implementation Test4
- (A*) test1 { return 0; } // id -> A* is rdar://problem/8596987
- (id) test2 { return 0; }
@end

// rdar://12522752
typedef int int32_t;
typedef long long int64_t;

@interface NSObject @end

@protocol CKMessage
@property (nonatomic,readonly,assign) int64_t sequenceNumber; // expected-note {{previous definition is here}}
@end

@protocol CKMessage;

@interface CKIMMessage : NSObject<CKMessage>
@end

@implementation CKIMMessage
- (int32_t)sequenceNumber { // expected-warning {{conflicting return type in implementation of 'sequenceNumber': 'int64_t' (aka 'long long') vs 'int32_t' (aka 'int')}}
  return 0;
}
@end

// rdar://14650159
// Tests that property inherited indirectly from a nested protocol
// is seen by the method implementation type matching logic before
// method in super class is seen. This fixes the warning coming
// out of that method mismatch.
@interface NSObject (NSDict)
- (void)setValue:(id)value;
- (id)value;
@end

@protocol ProtocolWithValue
@property (nonatomic) unsigned value;
@end

@protocol InterveningProtocol <ProtocolWithValue>
@end

@interface UsesProtocolWithValue : NSObject <ProtocolWithValue>
@end

@implementation UsesProtocolWithValue
@synthesize value=_value;
- (unsigned) value
{
	return _value;
}
- (void) setValue:(unsigned)value
{
	_value = value;
}
@end


@interface UsesInterveningProtocol : NSObject <InterveningProtocol>
@end

@implementation UsesInterveningProtocol

@synthesize value=_value;
- (unsigned) value
{
	return _value;
}
- (void) setValue:(unsigned)value
{
	_value = value;
}
@end
