// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

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
