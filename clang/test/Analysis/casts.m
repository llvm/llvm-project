// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -verify %s
// expected-no-diagnostics

// Test function pointer casts.
typedef void* (*MyFuncTest1)(void);

MyFuncTest1 test1_aux(void);
void test1(void) {
  void *x;
  void* (*p)(void);
  p = ((void*) test1_aux());
  if (p != ((void*) 0)) x = (*p)();
}

// Test casts from void* to function pointers.
void* test2(void *p) {
  MyFuncTest1 fp = (MyFuncTest1) p;
  return (*fp)();
}

// A cast from int onjective C property reference to int.
typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {} - (id)init; @end
typedef enum {
  EEOne,
  EETwo
} RDR10087620Enum;
@interface RDR10087620 : NSObject {
  RDR10087620Enum   elem;
}
@property (readwrite, nonatomic) RDR10087620Enum elem;
@end

static void
adium_media_ready_cb(RDR10087620 *InObj)
{
  InObj.elem |= EEOne;
}


// PR16690
_Bool testLocAsIntegerToBool(void) {
  return (long long)&testLocAsIntegerToBool;
}
