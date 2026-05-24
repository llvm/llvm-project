// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSObject @end

@interface DBGViewDebuggerSupport : NSObject
+ (void)addViewLayerInfo:(id)view;
- (void)addInstViewLayerInfo:(id)view;
@end

@interface DBGViewDebuggerSupport_iOS : DBGViewDebuggerSupport
@end

@implementation DBGViewDebuggerSupport_iOS
+ (void)addViewLayerInfo:(id)aView;
{
    [super addViewLayerInfo:view]; // expected-error {{use of undeclared identifier 'view'}}
}
- (void)addInstViewLayerInfo:(id)aView;
{
    [super addInstViewLayerInfo:view]; // expected-error {{use of undeclared identifier 'view'}}
}
@end
