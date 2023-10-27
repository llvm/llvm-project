// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

@interface NSObject @end

@interface Foo : NSObject
@property int p __attribute__((section("__TEXT,foo")));
@end

@implementation Foo @end

// CHECK: define internal i32 @"\01-[Foo p]"({{.*}} section "__TEXT,foo" {
// CHECK: define internal void @"\01-[Foo setP:]"({{.*}} section "__TEXT,foo" {
