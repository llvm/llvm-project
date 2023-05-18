// RUN: %clang_cc1 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

@protocol P1

@property int prop;

@end

@interface I <P1>

@end

@implementation I
@end // CHECK: fix-it:{{.*}}:{[[@LINE]]:1-[[@LINE]]:1}:"@synthesize prop;\n\n"
