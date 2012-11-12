// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core,deadcode.DeadStores -analyzer-store=region -analyzer-constraints=range -verify %s

// These declarations were reduced using Delta-Debugging from Foundation.h
// on Mac OS X.  The test cases are below.

typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef unsigned int NSUInteger;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
@end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
  + (id)alloc;
@end
typedef float CGFloat;
typedef struct _NSPoint {} NSRect;
NSRect NSMakeRect(CGFloat x, CGFloat y, CGFloat w, CGFloat h);
enum { NSBackingStoreRetained = 0,     NSBackingStoreNonretained = 1,     NSBackingStoreBuffered = 2 };
typedef NSUInteger NSBackingStoreType;
@interface NSResponder : NSObject <NSCoding> {}
@end
@protocol NSAnimatablePropertyContainer
- (id)animator;
@end
extern NSString *NSAnimationTriggerOrderIn ;
@class CIFilter, CALayer, NSDictionary, NSScreen, NSShadow, NSTrackingArea;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {} @end
@protocol NSValidatedUserInterfaceItem - (SEL)action; @end
@protocol NSUserInterfaceValidations - (BOOL)validateUserInterfaceItem:(id <NSValidatedUserInterfaceItem>)anItem; @end   @class NSNotification, NSText, NSView, NSMutableSet, NSSet, NSDate;
enum { NSBorderlessWindowMask = 0,     NSTitledWindowMask = 1 << 0,     NSClosableWindowMask = 1 << 1,     NSMiniaturizableWindowMask = 1 << 2,     NSResizableWindowMask = 1 << 3  };
@interface NSWindow : NSResponder  <NSAnimatablePropertyContainer, NSUserInterfaceValidations>    {
  struct __wFlags {} _wFlags;
}
- (id)initWithContentRect:(NSRect)contentRect styleMask:(NSUInteger)aStyle backing:(NSBackingStoreType)bufferingType defer:(BOOL)flag;
- (id)initWithContentRect:(NSRect)contentRect styleMask:(NSUInteger)aStyle backing:(NSBackingStoreType)bufferingType defer:(BOOL)flag screen:(NSScreen *)screen;
- (void)orderFrontRegardless;
@end

extern NSString *NSWindowDidBecomeKeyNotification;

// Test cases.

void f1() {
  NSWindow *window = [[NSWindow alloc]
                      initWithContentRect:NSMakeRect(0,0,100,100) 
                        styleMask:NSTitledWindowMask|NSClosableWindowMask
                        backing:NSBackingStoreBuffered
                        defer:0]; 

  [window orderFrontRegardless]; // no-warning
}

void f2() {
  NSWindow *window = [[NSWindow alloc]
                      initWithContentRect:NSMakeRect(0,0,100,100) 
                        styleMask:NSTitledWindowMask|NSClosableWindowMask
                        backing:NSBackingStoreBuffered
                        defer:0
                        screen:0]; 

  [window orderFrontRegardless]; // no-warning
}

void f2b() {
  // FIXME: NSWindow doesn't own itself until it is displayed.
  NSWindow *window = [[NSWindow alloc] // no-warning
                      initWithContentRect:NSMakeRect(0,0,100,100) 
                        styleMask:NSTitledWindowMask|NSClosableWindowMask
                        backing:NSBackingStoreBuffered
                        defer:0
                        screen:0]; 

  [window orderFrontRegardless];
  
  [window retain];
}


void f3() {
  // FIXME: For now we don't track NSWindow.
  NSWindow *window = [NSWindow alloc];  // expected-warning{{never read}}
}
