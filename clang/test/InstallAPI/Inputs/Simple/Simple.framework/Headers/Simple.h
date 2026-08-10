#import <Foundation/Foundation.h>

// Useless forward declaration. This is used for testing.
@class FooBar;
@protocol FooProtocol;

@protocol ForwardProcotol;

// Test public global.
extern int publicGlobalVariable;

// Test weak public global.
extern int weakPublicGlobalVariable __attribute__((weak));

// Test public ObjC class
@interface Simple : NSObject
@end

__attribute__((objc_exception))
@interface Base : NSObject
@end

@interface SubClass : Base
@end

@protocol BaseProtocol
- (void) baseMethod;
@end

NS_AVAILABLE(10_11, 9_0)
@protocol FooProtocol <BaseProtocol>
- (void) protocolMethod;
@end

@protocol BarProtocol
- (void) barMethod;
@end

@interface FooClass <FooProtocol, BarProtocol>
@end

// Create an empty category conforms to a forward declared protocol.
// <rdar://problem/35605892>
@interface FooClass (Test) <ForwardProcotol>
@end
