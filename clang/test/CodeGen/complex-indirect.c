// RUN: %clang_cc1 -emit-llvm %s -o %t -triple=x86_64-apple-darwin10
// RUN: FileCheck < %t %s

// Make sure this doesn't crash. We used to generate a byval here and wanted to
// verify a valid alignment, but we now realize we can use an i16 and let the
// backend guarantee the alignment.

void a(int,int,int,int,int,int,__complex__ char);
void b(__complex__ char *y) { a(0,0,0,0,0,0,*y); }
// CHECK: define void @b
// CHECK: alloca { i8, i8 }*, align 8
// CHECK: call void @a(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i16 {{.*}})
