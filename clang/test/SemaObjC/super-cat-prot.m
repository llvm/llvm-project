// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef signed char BOOL;
typedef unsigned int NSUInteger;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
typedef float CGFloat;
typedef struct _NSSize {} NSSize;
typedef struct _NSRect {} NSRect;
@interface NSResponder : NSObject <NSCoding> {} @end
@protocol NSAnimatablePropertyContainer - (id)animator; @end
extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {} @end
@class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView;
enum { NSBoxPrimary = 0, NSBoxSecondary = 1, NSBoxSeparator = 2, NSBoxOldStyle = 3, NSBoxCustom = 4};
typedef NSUInteger NSBoxType;
@interface NSBox : NSView {} - (NSBoxType)boxType; @end
@class NSArray, NSError, NSImage, NSView, NSNotificationCenter, NSURL;
@interface NSProBox:NSBox {} @end
enum IBKnobPosition { IBNoKnobPosition = -1, IBBottomLeftKnobPosition = 0, 
                      IBMiddleLeftKnobPosition, IBTopLeftKnobPosition,
                      IBTopMiddleKnobPosition, IBTopRightKnobPosition,
                      IBMiddleRightKnobPosition, IBBottomRightKnobPosition, 
                      IBBottomMiddleKnobPosition };
typedef enum IBKnobPosition IBKnobPosition;
typedef struct _IBInset {} IBInset;
@protocol IBObjectProtocol -(NSString *)inspectorClassName; @end
@protocol IBViewProtocol
  -(NSSize)minimumFrameSizeFromKnobPosition:(IBKnobPosition)position;
  -(IBInset)ibShadowInset;
@end
@class NSPasteboard;
@interface NSObject (NSObject_IBObjectProtocol) <IBObjectProtocol> @end
@interface NSView (NSView_IBViewProtocol) <IBViewProtocol>  - (NSRect)layoutRect; @end
typedef enum { NSProTextFieldSquareBezel = 0, NSProTextFieldRoundedBezel = 1, NSProTextFieldDisplayBezel = 2 } MKModuleReusePolicy;
@implementation NSProBox(IBAdditions)
-(NSString *)inspectorClassName { return 0; }
-(IBInset)ibShadowInset {
  if ([self boxType] == NSBoxSeparator) {
    return [super ibShadowInset];
  }
  while (1) {}
}
-(NSSize)minimumFrameSizeFromKnobPosition:(IBKnobPosition)knobPosition {
  if ([self boxType] != NSBoxSeparator)
    return [super minimumFrameSizeFromKnobPosition:knobPosition];
  while (1) {}
}
@end
