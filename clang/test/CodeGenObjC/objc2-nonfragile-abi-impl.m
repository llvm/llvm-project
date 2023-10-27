// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s

@interface Base @end

@interface Sub1 : Base @end

@implementation Sub1 @end

@implementation Base { 
@private 
  id ivar; 
} 
@end

