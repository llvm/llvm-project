// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end

@interface NSObject <NSObject> {} @end

typedef float CGFloat;

typedef enum { NSMinXEdge = 0, NSMinYEdge = 1, NSMaxXEdge = 2, NSMaxYEdge = 3 } NSFastEnumerationState;

@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end

@class NSString;

@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end

extern NSString * const NSBundleDidLoadNotification;

@interface NSObject(NSKeyValueObserving)
- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context;
- (void)removeObserver:(NSObject *)observer forKeyPath:(NSString *)keyPath;
@end

enum { NSCaseInsensitivePredicateOption = 0x01,     NSDiacriticInsensitivePredicateOption = 0x02 };

@interface NSResponder : NSObject <NSCoding> {}
@end

extern NSString * const NSFullScreenModeAllScreens;
@interface NSWindowController : NSResponder <NSCoding> {}
@end

extern NSString *NSAlignmentBinding ;

@interface _XCOQQuery : NSObject {}
@end

extern NSString *PBXWindowDidChangeFirstResponderNotification;

@interface PBXModule : NSWindowController {}
@end

@class _XCOQHelpTextBackgroundView;
@interface PBXOpenQuicklyModule : PBXModule
{
@private
  _XCOQQuery *_query;
}
@end

@interface PBXOpenQuicklyModule ()
@property(readwrite, retain) _XCOQQuery *query;
@end

@implementation PBXOpenQuicklyModule  
@synthesize query = _query;
- (void) _clearQuery
{
  [self.query removeObserver: self forKeyPath: @"matches"];
}
@end

