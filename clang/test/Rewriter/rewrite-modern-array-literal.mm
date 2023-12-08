// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

extern "C" void *sel_registerName(const char *);
@class NSString;

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

typedef unsigned long NSUInteger;

@interface NSArray 
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end

int i;
int main() {
  NSArray *array = @[ @"Hello", @1234 ];
  if (i) {
    NSArray *array = @[ @"Hello", @1234 ];
  }
  NSArray *array1 = @[ @"Hello", @1234, @[ @"Hello", @1234 ] ];
}

