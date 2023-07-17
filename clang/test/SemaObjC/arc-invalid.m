// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fblocks -Wno-objc-root-class -verify %s

// rdar://problem/10982793
// [p foo] in ARC creates a cleanup.
// The plus is invalid and causes the cleanup to go unbound.
// Don't crash.
@interface A
- (id) foo;
@end
void takeBlock(void (^)(void));
void test0(id p) {
  takeBlock(^{ [p foo] + p; }); // expected-error {{invalid operands to binary expression}}
}

void test1(void) {
  __autoreleasing id p; // expected-note {{'p' declared here}}
  takeBlock(^{ (void) p; }); // expected-error {{cannot capture __autoreleasing variable in a block}}
}

// rdar://17024681
@class WebFrame;
@interface WebView  // expected-note {{previous definition is here}}
- (WebFrame *)mainFrame;
@end

@interface WebView  // expected-error {{duplicate interface definition for class 'WebView'}}
@property (nonatomic, readonly, strong) WebFrame *mainFrame;
@end

@interface UIWebDocumentView
- (WebView *)webView;
@end

@interface UIWebBrowserView : UIWebDocumentView
@end

@interface StoreBanner @end

@implementation StoreBanner
+ (void)readMetaTagContentForUIWebBrowserView:(UIWebBrowserView *)browserView
{
  [[browserView webView] mainFrame];
}
@end
