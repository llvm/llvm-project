// RUN: %clang_cc1 -fsyntax-only -verify %s
// radar 7638810

@protocol NSObject @end

@interface NSObject <NSObject> @end

@interface UIResponder : NSObject
@end

@implementation UIResponder
@end

@interface UIView : UIResponder
@end

@implementation UIView
@end

@interface UIWebTiledView : UIView
@end

@implementation UIWebTiledView
@end

@interface UIWebDocumentView : UIWebTiledView
@end

@implementation UIWebDocumentView
@end

@interface UIWebBrowserView : UIWebDocumentView
@end

@implementation UIWebBrowserView
@end

@interface UIPDFView : UIView
@end

@implementation UIPDFView
@end

@interface UIWebPDFView : UIPDFView
@end

@implementation UIWebPDFView
@end

UIWebPDFView *getView()
{
    UIWebBrowserView *browserView;
    UIWebPDFView *pdfView;
    return pdfView ? pdfView : browserView; // expected-warning {{incompatible pointer types returning 'UIView<NSObject> *' from a function with result type 'UIWebPDFView *'}}
}
