// RUN: %clang_cc1 -fsyntax-only %s
// FIXME: This test case tests the patch applied in: http://lists.cs.uiuc.edu/pipermail/cfe-commits/Week-of-Mon-20080602/006017.html
//   Eventually that logic should be treated as an extension.

typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
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
@end

@class NSArray;

typedef struct {} NSFastEnumerationState;

@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end

@class NSString;

@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
- (id)objectAtIndex:(NSUInteger)index;
@end

typedef unsigned short unichar;

@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
@end

@interface NSSimpleCString : NSString
{}

@end

@interface NSConstantString : NSSimpleCString @end

extern void *_NSConstantStringClassReference;

@interface NSResponder : NSObject <NSCoding> {}
@end

@class NSDate, NSDictionary, NSError, NSException, NSNotification;

@interface NSWindowController : NSResponder <NSCoding> {}
@end

@class PBXBuildLog, PBXBuildLogItem, PBXBuildLogContainerItem, XCWorkQueueCommand, XCBuildLogContainerItemMutationState;

@protocol PBXBuildLogContainerItems <NSObject>
- (PBXBuildLog *)buildLog;
@end

@interface PBXBuildLogItem : NSObject {}
- (id <PBXBuildLogContainerItems>)superitem;
@end
@interface PBXBuildResultsModule
@end

@implementation PBXBuildResultsModule
- (void) revealItems
{
        PBXBuildLogItem *objItem;
        PBXBuildLogItem *superitem = [objItem superitem];
}
@end
