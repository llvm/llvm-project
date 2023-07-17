// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D_Bool=bool -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// rdar://13562505

@protocol OS_dispatch_object @end

@interface NSObject @end

@protocol OS_dispatch_queue <OS_dispatch_object> @end typedef NSObject<OS_dispatch_queue> *dispatch_queue_t;

typedef id<OS_dispatch_queue> dispatch_queue_i;
