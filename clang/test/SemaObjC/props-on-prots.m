// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef signed char BOOL;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL) isEqual:(id) object;
@end

@protocol NSCoding
- (void) encodeWithCoder:(NSCoder *) aCoder;
@end

@interface NSObject < NSObject > {} @end

typedef float CGFloat;

@interface NSResponder:NSObject < NSCoding > {} @end

@class XCElementView;

typedef struct _XCElementInset {} XCElementInset;

@protocol XCElementP < NSObject >
-(id) vertical;
@end

@protocol XCElementDisplayDelegateP;
@protocol XCElementTabMarkerP;

typedef NSObject < XCElementTabMarkerP > XCElementTabMarker;

@protocol XCElementTabberP < XCElementP >
-(void) setMarker:(XCElementTabMarker *) marker;
@end

typedef NSObject < XCElementTabberP > XCElementTabber;

@protocol XCElementTabMarkerP < NSObject >
@property(nonatomic)
BOOL variableSized;
@end

@protocol XCElementJustifierP < XCElementP >
-(void) setHJustification:(CGFloat) hJust;
@end

typedef NSObject < XCElementJustifierP > XCElementJustifier;
@interface XCElementImp:NSObject < XCElementP > {}
@end

@class XCElementImp;

@interface XCElementTabberImp:XCElementImp < XCElementTabberP > {
	XCElementTabMarker *_marker;
}
@end

@implementation XCElementTabberImp 
- (void) setMarker:(XCElementTabMarker *) marker {
  if (_marker && _marker.variableSized) {
  }
}
- (id)vertical { return self; }
- (BOOL)isEqual:x { return 1; }
@end
