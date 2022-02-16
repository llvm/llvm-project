// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s
// rdar://18961148

void syslog(const char *, ...);

void handler( );

__attribute__((used))
static void (^spd)(void) = ^(void)
{
 handler( ^(){ syslog("%s", __FUNCTION__); } );
};
// CHECK: @__FUNCTION__.spd_block_invoke_2 = private unnamed_addr constant [19 x i8] c"spd_block_invoke_2\00"
// CHECK: define internal void @spd_block_invoke_2
// CHECK: @__FUNCTION__.spd_block_invoke_2
