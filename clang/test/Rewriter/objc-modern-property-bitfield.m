// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp

void *sel_registerName(const char *);
extern void abort();

@interface NSObject 
+ alloc;
- init;
@end

typedef unsigned char BOOL;

@interface Foo : NSObject {

   BOOL  _field1 : 5;
   BOOL  _field2    : 3;
}

@property BOOL field1;
@property BOOL field2;
@end

@implementation Foo

@synthesize field1 = _field1;
@synthesize field2 = _field2;

@end

int main()
{
  Foo *f = (Foo*)[[Foo alloc] init];
  f.field1 = 0xF;
  f.field2 = 0x3;
  f.field1 = f.field1 & f.field2;
  if (f.field1 != 0x3)
    abort ();
  return 0; 
}


