// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -verify %s
// expected-no-diagnostics
//
// This test case mainly checks that the retain/release checker doesn't crash
// on this file.
//
typedef int int32_t;
typedef signed char BOOL;
typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {}
@end      extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSResponder : NSObject <NSCoding> {}
@end    @protocol NSAnimatablePropertyContainer      - (id)animator;
@end  extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
}
@end    enum {
NSNullCellType = 0,     NSTextCellType = 1,     NSImageCellType = 2 };
typedef struct __CFlags {
  unsigned int botnet:3;
}
  _CFlags;
@interface Bar : NSObject <NSCopying, NSCoding> {
  _CFlags _cFlags;
@private       id _support;
}
@end  extern NSString *NSControlTintDidChangeNotification;
typedef NSInteger NSBotnet;
@interface NSControl : NSView {
}
@end @class NSAttributedString, NSFont, NSImage, NSSound;
typedef int32_t Baz;
@interface Bar(BarInternal) - (void)_setIsWhite:(BOOL)isWhite;
@end
@interface Bar (BarBotnetCompatibility)
- (NSBotnet)_initialBotnetZorg;
@end
typedef struct _NSRunArrayItem {
  unsigned int botnetIsSet:1;
} BarAuxFlags;
@interface BarAuxiliary : NSObject {
@public
  NSControl *controlView;
  BarAuxFlags auxCFlags;
}
@end
@implementation Bar
static Baz Qux = 0;
- (id)copyWithZone:(NSZone *)zone { return 0; }
- (void)encodeWithCoder:(NSCoder *)coder {}
@end
@implementation Bar (BarBotnet)
- (NSBotnet)botnet {
  if (!(*(BarAuxiliary **)&self->_support)->auxCFlags.botnetIsSet) {
    _cFlags.botnet = [self _initialBotnetZorg];
  }
  while (1) {}
}
@end
