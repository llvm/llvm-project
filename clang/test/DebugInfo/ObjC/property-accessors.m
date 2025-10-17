// RUN: %clang_cc1 -emit-llvm -x objective-c -debug-info-kind=limited -triple x86_64-apple-macosx10.8.0 %s -o - | FileCheck %s
//
// Ensure we emit the names of explicit/renamed accessors even if they
// are defined later in the implementation section.
//
// CHECK:  !DIObjCProperty(name: "blah"
// CHECK-SAME:             getter: "isBlah"

@class NSString;
extern void NSLog(NSString *format, ...);
typedef signed char BOOL;

#define YES             ((BOOL)1)
#define NO              ((BOOL)0)

typedef unsigned int NSUInteger;

@protocol NSObject
@end

@interface NSObject <NSObject>
- (id)init;
+ (id)alloc;
@end

@interface Bar : NSObject
@property int normal_property;
@property (getter=isBlah, setter=setBlah:) BOOL blah;
@end

@implementation Bar
@synthesize normal_property;

- (BOOL) isBlah
{
  return YES;
}
- (void) setBlah: (BOOL) newBlah
{
  NSLog (@"Speak up, I didn't catch that.");
}
@end

int
main (void)
{
  Bar *my_bar = [[Bar alloc] init];

  if (my_bar.blah)
    NSLog (@"It was true!!!");

  my_bar.blah = NO;

  return 0;
}
