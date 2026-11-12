// RUN: %clang_cc1 -emit-llvm -o %t %s

__attribute__((visibility("hidden")))
@interface Hidden
+(void) bar;
@end

@implementation Hidden
+(void) bar {}
@end

__attribute__((visibility("default")))
@interface Default
+(void) bar;
@end

@implementation Default
+(void) bar {}
@end
