// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s

@interface NSObject
@end

@implementation NSObject(IBXLIFFIntegration)
@end

