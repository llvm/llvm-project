// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -analyzer-store=region -verify -Wno-objc-root-class %s
// expected-no-diagnostics

// BEGIN delta-debugging reduced header stuff

typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (oneway void)release;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)alloc;
@end
typedef float CGFloat;
typedef struct _NSPoint {} NSRect;
static __inline__ __attribute__((always_inline)) NSRect NSMakeRect(CGFloat x, CGFloat y, CGFloat w, CGFloat h) { NSRect r; return r; }
typedef struct {} NSFastEnumerationState;
@protocol NSFastEnumeration 
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSString;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end
@interface NSMutableArray : NSArray
- (void)addObject:(id)anObject;
@end @class NSAppleEventDescriptor;
enum { NSBackingStoreRetained = 0,     NSBackingStoreNonretained = 1,     NSBackingStoreBuffered = 2 };
typedef NSUInteger NSBackingStoreType;
@interface NSResponder : NSObject <NSCoding> {} @end
@protocol NSAnimatablePropertyContainer
- (id)animator;
@end
@protocol NSValidatedUserInterfaceItem
- (SEL)action;
@end
@protocol NSUserInterfaceValidations
- (BOOL)validateUserInterfaceItem:(id <NSValidatedUserInterfaceItem>)anItem;
@end  @class NSDate, NSDictionary, NSError, NSException, NSNotification;
enum { NSBorderlessWindowMask = 0,     NSTitledWindowMask = 1 << 0,     NSClosableWindowMask = 1 << 1,     NSMiniaturizableWindowMask = 1 << 2,     NSResizableWindowMask = 1 << 3  };
@interface NSWindow : NSResponder  <NSAnimatablePropertyContainer, NSUserInterfaceValidations>    {}
- (id)initWithContentRect:(NSRect)contentRect styleMask:(NSUInteger)aStyle backing:(NSBackingStoreType)bufferingType defer:(BOOL)flag;
@end
extern NSString *NSWindowDidBecomeKeyNotification;
@interface NSPanel : NSWindow {}
@end
@class NSTableHeaderView;

// END delta-debugging reduced header stuff

@interface MyClass
{
	NSMutableArray *panels;
}
- (void)myMethod;
- (void)myMethod2;
@end

@implementation MyClass // no-warning
- (void)myMethod
{
  NSPanel *panel = [[NSPanel alloc] initWithContentRect:NSMakeRect(0, 0, 200, 200) styleMask:NSBorderlessWindowMask backing:NSBackingStoreBuffered defer:(BOOL)1];

  [panels addObject:panel];

  [panel release]; // no-warning
}
- (void)myMethod2
{
  NSPanel *panel = [[NSPanel alloc] initWithContentRect:NSMakeRect(0, 0, 200, 200) styleMask:NSBorderlessWindowMask backing:NSBackingStoreBuffered defer:(BOOL)1]; // no-warning

  [panels addObject:panel];  
}
@end

