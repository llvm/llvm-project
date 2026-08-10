// RUN: %clang_cc1 -fsyntax-only -verify %s
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject - (BOOL) isEqual:(id) object; - (id)init; @end
@protocol NSCopying - (id) copyWithZone:(NSZone *) zone; @end
@protocol NSCoding - (void) encodeWithCoder:(NSCoder *) aCoder; @end

@interface NSObject < NSObject > {}
+(id) alloc;
@end

typedef float CGFloat;

@interface NSTask:NSObject
- (id) init;
@end

typedef NSUInteger NSControlSize;
typedef struct __CFlags {} _CFlags;

@interface NSCell:NSObject < NSCopying, NSCoding > {}
@end

@interface NSActionCell:NSCell {} @end

@class NSAttributedString, NSFont, NSImage, NSSound;

typedef struct _XCElementInset {} XCElementInset;

@protocol XCElementP < NSObject >
-(BOOL) vertical;
@end

@protocol XCElementDisplayDelegateP;
@protocol XCElementDisplayDelegateP < NSObject >
-(void) configureForControlSize:(NSControlSize)size font:(NSFont *)font addDefaultSpace:(XCElementInset) additionalSpace;
@end

@protocol XCElementSpacerP < XCElementP >
@end

typedef NSObject < XCElementSpacerP > XCElementSpacer;

@protocol XCElementTogglerP < XCElementP > -(void) setDisplayed:(BOOL) displayed;
@end

typedef NSObject < XCElementTogglerP > XCElementToggler; // expected-note {{previous definition is here}}

@interface XCElementRootFace:NSObject {} @end

@interface XCElementFace:XCElementRootFace {} @end

@class XCElementToggler;  // expected-warning {{redefinition of forward class 'XCElementToggler' of a typedef name of an object type is ignored}}

@interface XCRASlice:XCElementFace {} @end

@class XCElementSpacings;

@interface XCElementDisplay:NSObject < XCElementDisplayDelegateP > {} @end
@interface XCElementDisplayRect:XCElementDisplay {} @end

typedef XCElementDisplayRect XCElementGraphicsRect;

@interface XCElementDisplayFillerImage:XCElementDisplay {} @end

@implementation XCRASlice
- (void) addSliceWithLabel:(NSString *)label statusKey:(NSString *)statusKey disclosed:(BOOL)disclosed
{
  static XCElementGraphicsRect *_sGraphicsDelegate = ((void *) 0);
  if (!_sGraphicsDelegate) {
    _sGraphicsDelegate =[[XCElementGraphicsRect alloc] init]; 
  }
}
@end
