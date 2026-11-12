// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface PDFViewPrivateVars 
{
@public
	__attribute__((objc_gc(strong))) char *addedTooltips;
}
@end

@interface PDFView 
{
    PDFViewPrivateVars *_pdfPriv;
}
@end

@implementation PDFView
- (void) addTooltipsForPage
{
 _pdfPriv->addedTooltips[4] = 1;
}
@end

