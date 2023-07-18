// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-apple-darwin9 -Wno-objc-root-class %s
// expected-no-diagnostics
// FIXME: must also compile as Objective-C++ 

// <rdar://problem/6487662>
typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (BOOL)respondsToSelector:(SEL)aSelector;
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
@class NSString, NSData;
typedef struct _NSPoint {}
NSRange;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
@end
@interface NSMutableString : NSString
- (void)replaceCharactersInRange:(NSRange)range withString:(NSString *)aString;
@end
@class NSArray, NSDictionary, NSMapTable;
@interface NSResponder : NSObject <NSCoding> {}
@end
@protocol NSAnimatablePropertyContainer
- (id)animator;
@end
extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
  struct __VFlags2 {} _vFlags2;
}
@end
@class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView;
@interface FooiagramView : NSView {
id _delegate;
}
@end
@class FooiagramView;
@interface _FooiagramViewReserved : NSObject {
@public
  NSMutableString *_typeToSelectString;
  struct _FooiagramViewFlags {
      unsigned int delegateRespondsToPrintInfoForBarView : 1;
  } _dvFlags;
}
@end
extern _FooiagramViewReserved *_FooiagramViewBarViewReserved(FooiagramView *BarView);
@interface FooiagramView (FooiagramViewPrivate)
+ (Class)_defaultBarToolManagerClass;
@end
@implementation FooiagramView
static NSMapTable *_defaultMenuForClass = 0;
- (void)setDelegate:(id)delegate {
  if (_delegate != delegate) {
    struct _FooiagramViewFlags *dvFlags =
      &_FooiagramViewBarViewReserved(self)->_dvFlags;
    if (_delegate != ((void *)0)) {
      dvFlags->delegateRespondsToPrintInfoForBarView = [_delegate respondsToSelector:@selector(printInfoForBarView:)];
    }
  }
}
@end

// <rdar://problem/6487684>
@interface WizKing_MIKeep {
struct __LoreStuffNode *_historyStuff;
}
@end
typedef struct __LoreStuffNode {} LoreStuffNode;
@implementation WizKing_MIKeep
- init {
  LoreStuffNode *node;
  node = &(_historyStuff[1]);
  return 0;
}
@end

// <rdar://problem/6487702>
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
void *memset(void *, int, size_t);
@class NSString, NSURL, NSError;
@interface OingoWerdnaPeon : NSObject {}
@end        typedef enum {
OingoPT_SmashOK,     OingoPT_NoSuchFile, }
OingoWerdnaPeonIOMethod;
@interface OingoWerdnaPeonSmashDrivel : NSObject <NSCopying> {}
@end
@interface OingoBoingoContraptionPeon : OingoWerdnaPeon {
struct _OingoBoingoContraptionPeonFlags {}
_nfttFlags;
}
@end
@implementation OingoBoingoContraptionPeon
+ (void)initialize {}
- (id)initWithSmashDrivel:(OingoWerdnaPeonSmashDrivel *)info {
  if (self != ((void *)0)) {
    (void)memset(&_nfttFlags, 0, sizeof(struct _OingoBoingoContraptionPeonFlags));
  }
  return 0;
}
@end

@interface Blah {
  struct X {
    int x;
  } value;
}
@end

@implementation Blah
- (int)getValue {
  struct X *xp = &value;
  return xp->x;
}
@end
