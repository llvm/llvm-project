// RUN: %clang_cc1 -triple i386-unknown-unknown -x c++ -emit-llvm -o - %s | FileCheck %s

union _u { char a[]; } u = {};
union _u0 { char a[]; } u0 = {0};

// CHECK: %union._u = type { [0 x i8] }

// CHECK: @u = global %union._u zeroinitializer, align 1
// CHECK: @u0 = global { [1 x i8] } zeroinitializer, align 1

union { char a[]; } z = {};
// CHECK: @z = internal global %union.{{.*}} zeroinitializer, align 1
union { char a[]; } z0 = {0};
// CHECK: @z0 = internal global { [1 x i8] } zeroinitializer, align 1

/* C++ requires global anonymous unions have static storage, so we have to
   reference them to keep them in the IR output. */
char keep(int pick)
{
	if (pick)
		return z.a[0];
	else
		return z0.a[0];
}
