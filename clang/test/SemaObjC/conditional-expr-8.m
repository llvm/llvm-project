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

