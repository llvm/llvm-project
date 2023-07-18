// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -print-ivar-layout -emit-llvm %s -o %t-64.s | FileCheck -check-prefix CHECK-LP64 %s
// rdar://8991729

@interface NSObject {
  id isa;
}
@end

@interface AllPointers : NSObject {
    id foo;
    id __strong bar;    
    NSObject *bletch;
}
@end

@implementation AllPointers
@end
// CHECK-LP64: strong ivar layout for class 'AllPointers': 0x03, 0x00

@class NSString, NSNumber;
@interface A : NSObject {
   NSString *foo;
   NSNumber *bar;
   unsigned int bletch;
   __weak id delegate;
}
@end

@interface B : A {
  unsigned int x;
  NSString *y;
  NSString *z;
}
@end

@implementation A @end

// CHECK-LP64: strong ivar layout for class 'A': 0x02, 0x00
// CHECK-LP64: weak ivar layout for class 'A': 0x31, 0x00

@implementation B @end

// CHECK-LP64: strong ivar layout for class 'B': 0x12, 0x00

@interface UnsafePerson {
@public
    __unsafe_unretained id name;
    __unsafe_unretained id age;
    id value;
}
@end

@implementation UnsafePerson @end

// CHECK-LP64: strong ivar layout for class 'UnsafePerson': 0x21, 0x00

// rdar://16136439
@interface rdar16136439
    @property (nonatomic, readonly, weak) id first;
@end

@implementation rdar16136439 @end

// CHECK-LP64: weak ivar layout for class 'rdar16136439': 0x01, 0x00

@interface Misalign : NSObject {
  char a;
}
@end

@interface Misaligned : Misalign {
  char b;
  id x;
}
@end

@implementation Misaligned @end

// CHECK-LP64: strong ivar layout for class 'Misaligned': 0x01, 0x00
