// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// rdar://5967199

typedef signed char BOOL;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL) isEqual:(id) object;
@end

@protocol NSCoding
- (void) encodeWithCoder:(NSCoder *) aCoder;
@end

@interface NSObject < NSObject > {}
@end

typedef float CGFloat;
typedef struct _NSPoint {} NSSize;
typedef struct _NSRect {} NSRect;
typedef enum { NSMinXEdge = 0, NSMinYEdge = 1, NSMaxXEdge = 2, NSMaxYEdge = 3} NSRectEdge;
extern void NSDivideRect(NSRect inRect, NSRect * slice, NSRect * rem, CGFloat amount, NSRectEdge edge);

@interface NSResponder:NSObject < NSCoding > {}
@end

@protocol NSAnimatablePropertyContainer
- (id) animator;
@end

extern NSString *NSAnimationTriggerOrderIn;

@interface NSView:NSResponder < NSAnimatablePropertyContainer > {}
-(NSRect) bounds;
@end

enum {
  NSBackgroundStyleLight = 0, NSBackgroundStyleDark, NSBackgroundStyleRaised, NSBackgroundStyleLowered
};

@interface NSTabView:NSView {}
@end

@ class OrganizerTabHeader;

@interface OrganizerTabView:NSTabView {}
@property(assign)
NSSize minimumSize;
@end

@interface OrganizerTabView()
@property(readonly) OrganizerTabHeader *tabHeaderView;
@property(readonly) NSRect headerRect;
@end

@implementation OrganizerTabView
@dynamic tabHeaderView, headerRect, minimumSize;
-(CGFloat) tabAreaThickness { return 0; }
-(NSRectEdge) rectEdgeForTabs { 
  NSRect dummy, result = {};
  NSDivideRect(self.bounds, &result, &dummy, self.tabAreaThickness, self.rectEdgeForTabs);
  return 0;
}
@end

@class NSImage;

@interface XCImageArchiveEntry : NSObject
{
  NSImage *_cachedImage;
}

@end

@implementation XCImageArchiveEntry

- (NSImage *)image
{
  return _cachedImage;
}

@end

@interface XCImageArchive : NSObject
@end

@implementation XCImageArchive

- (NSImage *)imageNamed:(NSString *)name
{
    XCImageArchiveEntry * entry;
    return entry ? entry.image : ((void *)0);
}

@end
