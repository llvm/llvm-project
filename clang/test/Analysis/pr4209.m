// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9 -analyzer-checker=core,alpha.core -Wno-incomplete-implementation -verify %s

// This test case was crashing due to how CFRefCount.cpp resolved the
// ObjCInterfaceDecl* and ClassName in EvalObjCMessageExpr.

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {
}
@end  typedef float CGFloat;
typedef struct _NSPoint {
}
NSFastEnumerationState;
@protocol NSFastEnumeration  - (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end        @class NSString;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end    @interface NSMutableArray : NSArray  - (void)addObject:(id)anObject;
@end         typedef unsigned short unichar;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
- (int)intValue;
@end @interface NSSimpleCString : NSString {
}
@end  @interface NSConstantString : NSSimpleCString @end   extern void *_NSConstantStringClassReference;
@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end    @interface NSMutableDictionary : NSDictionary  - (void)removeObjectForKey:(id)aKey;
@end       typedef struct {
}
CMProfileLocation;
@interface NSResponder : NSObject <NSCoding> {
}
@end  @class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView;
@interface NSCell : NSObject <NSCopying, NSCoding> {
}
@end  extern NSString *NSControlTintDidChangeNotification;
@interface NSActionCell : NSCell {
}
@end  @class NSArray, NSDocument, NSWindow;
@interface NSWindowController : NSResponder <NSCoding> {
}
@end         @class EBayCategoryType, GSEbayCategory, GBSearchRequest;
@interface GBCategoryChooserPanelController : NSWindowController {
  GSEbayCategory *rootCategory;
}
- (NSMutableDictionary*)categoryDictionaryForCategoryID:(int)inID inRootTreeCategories:(NSMutableArray*)inRootTreeCategories;
-(NSString*) categoryID; // expected-note {{using}}
@end @interface GSEbayCategory : NSObject <NSCoding> {
}
- (int) categoryID; // expected-note {{also found}}
- (GSEbayCategory *) parent;
- (GSEbayCategory*) subcategoryWithID:(int) inID;
@end   @implementation GBCategoryChooserPanelController  + (int) chooseCategoryIDFromCategories:(NSArray*) inCategories        searchRequest:(GBSearchRequest*)inRequest         parentWindow:(NSWindow*) inParent {
  return 0;
}
- (void) addCategory:(EBayCategoryType*)inCategory toRootTreeCategory:(NSMutableArray*)inRootTreeCategories {
  GSEbayCategory *category = [rootCategory subcategoryWithID:[[inCategory categoryID] intValue]]; // expected-warning {{multiple methods named 'categoryID' found}}

  if (rootCategory != category)  {
    GSEbayCategory *parent = category;
    while ((((void*)0) != (parent = [parent parent])) && ([parent categoryID] != 0))   {
      NSMutableDictionary *treeCategoryDict = [self categoryDictionaryForCategoryID:[parent categoryID] inRootTreeCategories:inRootTreeCategories];
      if (((void*)0) == treeCategoryDict)    {
      }
    }
  }
}
@end
