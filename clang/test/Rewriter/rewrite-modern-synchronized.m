// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions  -Wno-address-of-temporary -D"SEL=void*" -D"Class=struct objc_class *" -D"__declspec(X)=" %t-rw.cpp

typedef struct objc_object {
    Class isa;
} *id;

void *sel_registerName(const char *);

id SYNCH_EXPR(void);
void SYNCH_BODY(void);
void  SYNCH_BEFORE(void);
void  SYNC_AFTER(void);

void foo(id sem)
{
  SYNCH_BEFORE();
  @synchronized (SYNCH_EXPR()) { 
    SYNCH_BODY();
    return;
  }
 SYNC_AFTER();
 @synchronized ([sem self]) {
    SYNCH_BODY();
    return;
 }
}

void test_sync_with_implicit_finally(void) {
    id foo;
    @synchronized (foo) {
        return; // The rewriter knows how to generate code for implicit finally
    }
}

// rdar://14993814
@interface NSObject @end

@interface I : NSObject @end

@implementation I
+ (void) Meth {
@synchronized(self) {
}
}
@end
