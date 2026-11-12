// RUN: %clang -S -emit-llvm %s -o - -x objective-c -fobjc-runtime=gcc | FileCheck --check-prefix=GCC %s
// RUN: %clang -S -emit-llvm %s -o - -x objective-c -fobjc-runtime=gnustep-1.5 | FileCheck --check-prefix=GCC %s
// RUN: %clang -S -emit-llvm %s -o - -x objective-c -fobjc-runtime=gnustep-1.6 | FileCheck --check-prefix=GNUSTEP %s
//
@interface helloclass  {
@private int varName;
}
@property (readwrite,assign) int propName;
@end

@implementation helloclass
@synthesize propName = varName;
@end
// GCC-NOT: Ti,VvarName
// GNUSTEP: Ti,VvarName
