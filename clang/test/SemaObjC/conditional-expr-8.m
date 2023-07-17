// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// rdar://9296866
@interface NSResponder
@end


@interface NSView : NSResponder
@end

@interface WebView : NSView
@end

@protocol WebDocumentView
@end

@implementation NSView

- (void) FUNC : (id)s {
  WebView *m_webView;
  NSView <WebDocumentView> *documentView;
  NSView *coordinateView = s ?  documentView : m_webView;
}
@end

// rdar://problem/19572837
@protocol NSObject
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@protocol Goable <NSObject>
- (void)go;
@end

@protocol Drivable <Goable>
- (void)drive;
@end

@interface Car : NSObject
- (NSObject <Goable> *)bestGoable:(NSObject <Goable> *)drivable;
@end

@interface Car(Category) <Drivable>
@end

@interface Truck : Car
@end

@implementation Truck
- (NSObject <Goable> *)bestGoable:(NSObject <Goable> *)drivable value:(int)value{
    return value > 0 ? self : drivable;
}
@end
