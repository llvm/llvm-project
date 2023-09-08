// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -D"id=void*" -D"__declspec(X)=" %t-rw.cpp

@protocol NSPortDelegate;
@interface NSConnection @end

@interface NSMessagePort
- (void) clone;
@end

@implementation NSMessagePort
- (void) clone {
     NSConnection <NSPortDelegate> *conn = 0;
     id <NSPortDelegate> *idc = 0;
}
@end

@protocol Proto1, Proto2;

@protocol Proto
@end

unsigned char func(id<Proto1, Proto2> inProxy);

id bar(id);

void f() {
        id a;
        id b = bar((id <Proto>)a);
}

@protocol NSObject @end
@class NSRunLoop;

@protocol CoreDAVTaskManager <NSObject> 
  @property (retain) NSRunLoop *workRunLoop;  
@end

@protocol some_protocol;

void foo (int n)
{
  id<some_protocol> array[n];
}

