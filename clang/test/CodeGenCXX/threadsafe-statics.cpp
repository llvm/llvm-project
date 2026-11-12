// RUN: %clang_cc1 -emit-llvm -triple=x86_64-apple-darwin10 -o - %s | FileCheck -check-prefix=WITH-TSS %s
// RUN: %clang_cc1 -emit-llvm -triple=x86_64-apple-darwin10 -o - %s -fno-threadsafe-statics | FileCheck -check-prefix=NO-TSS %s

int f();

// WITH-TSS: @_ZZ1gvE1a = internal global i32 0, align 4
// WITH-TSS: @_ZGVZ1gvE1a = internal global i64 0

// WITH-TSS: define{{.*}} void @_Z1gv() [[NUW:#[0-9]+]]
// WITH-TSS: call i32 @__cxa_guard_acquire
// WITH-TSS: call void @__cxa_guard_release
// WITH-TSS: ret void
void g() { 
  static int a = f();
}

// NO-TSS: @_ZZ1gvE1a = internal global i32 0, align 4
// NO-TSS: @_ZGVZ1gvE1a = internal global i8 0

// NO-TSS: define{{.*}} void @_Z1gv() [[NUW:#[0-9]+]]
// NO-TSS-NOT: call i32 @__cxa_guard_acquire
// NO-TSS-NOT: call void @__cxa_guard_release
// NO-TSS: ret void

// WITH-TSS: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }

// NO-TSS: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
