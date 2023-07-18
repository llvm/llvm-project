// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://20350364

@interface NSObject @end

@interface DBGViewDebuggerSupport : NSObject
+ (void)addViewLayerInfo:(id)view;
- (void)addInstViewLayerInfo:(id)view;
@end

@interface DBGViewDebuggerSupport_iOS : DBGViewDebuggerSupport
@end

@implementation DBGViewDebuggerSupport_iOS
+ (void)addViewLayerInfo:(id)aView; // expected-note {{'aView' declared here}}
{
    [super addViewLayerInfo:view]; // expected-error {{use of undeclared identifier 'view'; did you mean 'aView'?}}
}
- (void)addInstViewLayerInfo:(id)aView; // expected-note {{'aView' declared here}}
{
    [super addInstViewLayerInfo:view]; // expected-error {{use of undeclared identifier 'view'; did you mean 'aView'?}}
}
@end
