// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

void *sel_registerName(const char *);

@interface  NSURLDownload
-(void)setBool:(bool)Arg;
@end

@implementation NSURLDownload
- (void) Meth
{
   [self setBool:(signed char)1];
}
@end

