// RUN: %clang_cc1 -analyzer-checker=core -verify %s -Wno-incomplete-implementation
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {
}
@end    extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value;
@end       @class NSString, NSData, NSMutableData, NSMutableDictionary, NSMutableArray;
typedef struct {
}
  NSFastEnumerationState;
@protocol NSFastEnumeration  - (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end        @class NSString;
typedef struct _NSRange {
}
  NSRange;
@interface NSValue (NSValueRangeExtensions)  + (NSValue *)valueWithRange:(NSRange)range;
- (id)objectAtIndex:(NSUInteger)index;
@end         typedef unsigned short unichar;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
@end       @class NSArray, NSDictionary, NSString, NSError;
@interface NSSet : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end    extern NSString *NSAccessibilityRoleDescription(NSString *role, NSString *subrole)     ;
@interface NSResponder : NSObject <NSCoding> {
}
@end    @protocol NSAnimatablePropertyContainer      - (id)animator;
@end  extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
}
@end  @class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView;
@interface NSWindowController : NSResponder <NSCoding> {
}
@end @class NSArray, NSFont, NSTabViewItem;
@interface NSTabView : NSView {
}
- (NSArray *)tabViewItems;
- (NSString *)label;
@end typedef enum {
PBXNoItemChanged = 0x00,     PBXProjectItemChanged = 0x01,     PBXReferenceChanged = 0x02,     PBXGroupChanged = 0x04,     PBXTargetChanged = 0x08,     PBXBuildPhaseChanged = 0x10,     PBXBuildFileChanged = 0x20,     PBXBreakpointChanged = 0x40, }
  PBXArchiveMask;
@interface PBXModule : NSWindowController {
}
@end       typedef enum {
PBXFindMatchContains,     PBXFindMatchStartsWith,     PBXFindMatchWholeWords,     PBXFindMatchEndsWith }
  PBXFindMatchStyle;
@protocol PBXSelectableText  - (NSString *)selectedString;
@end  @protocol PBXFindableText <PBXSelectableText>    - (BOOL)findText:(NSString *)string ignoreCase:(BOOL)ignoreCase matchStyle:(PBXFindMatchStyle)matchStyle backwards:(BOOL)backwards wrap:(BOOL)wrap;
@end  @class PBXProjectDocument, PBXProject, PBXAttributedStatusView;
@interface PBXProjectModule : PBXModule <PBXFindableText> {
}
@end @class PBXBookmark;
@protocol PBXSelectionTarget - (NSObject <PBXSelectionTarget> *) performAction:(id)action withSelection:(NSArray *)selection;  // expected-note {{method 'performAction:withSelection:' declared here}}
@end @class XCPropertyDictionary, XCPropertyCondition, XCPropertyConditionSet, XCMutablePropertyConditionSet;
extern NSMutableArray *XCFindPossibleKeyModules(PBXModule *module, BOOL useExposedModulesOnly);
@interface NSString (StringUtilities) - (NSString *) trimToLength:(NSInteger)length preserveRange:(NSRange)range;
- (id) objectOfType:(Class)type matchingFunction:(BOOL (void *, void *))comparator usingData:(void *)data;
@end  @class XCControlView;
@protocol XCDockViewHeader - (NSImage *) headerImage;
@end  @class XCDockableTabModule;
@interface XCExtendedTabView : NSTabView <XCDockViewHeader> {
}
@end     @class PBXProjectDocument, PBXFileReference, PBXModule, XCWindowTool;
@interface XCPerspectiveModule : PBXProjectModule <PBXSelectionTarget> {
  XCExtendedTabView *_perspectivesTabView;
}
- (PBXModule *) moduleForTab:(NSTabViewItem *)item;
@end  
@implementation XCPerspectiveModule // expected-warning {{method 'performAction:withSelection:' in protocol 'PBXSelectionTarget' not implemented}}}
+ (void) openForProjectDocument:(PBXProjectDocument *)projectDocument {
}
- (PBXModule *) type:(Class)type inPerspective:(id)perspectiveIdentifer  matchingFunction:(BOOL (void *, void *))comparator usingData:(void *)data {
  NSArray *allItems = [_perspectivesTabView tabViewItems];
  NSInteger i, c = [allItems count];
  for (i = 0;
       i < c;
       i++) {
    NSTabViewItem *item = [allItems objectAtIndex:i];
    if ([[item label] isEqual:perspectiveIdentifer]) {
      PBXProjectModule *pModule = (PBXProjectModule *)[self moduleForTab:item];
      PBXModule *obj = [XCFindPossibleKeyModules(pModule, (BOOL)0) objectOfType:type     matchingFunction:comparator usingData:data];
    }
  }
  return 0;
}
- (BOOL)buffer:(char *)buf containsAnyPrompts:(char *[])prompts
{
 prompts++;
 return (BOOL)0;
}
@end
