// RUN: %clang_cc1 -E %s -o %t.m
// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %t.m -o %t-rw.cpp 
// RUN: FileCheck --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

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

// CHECK: (*(NSURLResponse **)((char *)(*(NSURLResponse **)((char *)(*(NSCachedURLResponseInternal **)((char *)self + OBJC_IVAR_$_NSCachedURLResponse$_internal)) + OBJC_IVAR_$_NSCachedURLResponseInternal$response)) + OBJC_IVAR_$_NSURLResponse$InnerResponse)) = 0;
