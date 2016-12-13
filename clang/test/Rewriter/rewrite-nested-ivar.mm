// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw-modern.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw-modern.cpp
// radar 7583971


@interface NSURLResponse {
@public
  NSURLResponse *InnerResponse;
}
@end

@interface NSCachedURLResponseInternal 
{
    @public
    NSURLResponse *response;
}
@end

@interface NSCachedURLResponse
{
    @private
    NSCachedURLResponseInternal *_internal;
}
- (void) Meth;
@end

@implementation NSCachedURLResponse
- (void) Meth {
    _internal->response->InnerResponse = 0;
  }
@end
