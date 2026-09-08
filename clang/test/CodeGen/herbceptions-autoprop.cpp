// RUN: %clang -fherbceptions -fno-exceptions -S -emit-llvm -o - %s | FileCheck %s

// In C++, a bare call to a throws function inside a throws function
// auto-propagates the error (no explicit try() needed).

// CHECK: define dso_local { i32, i1 } @_Z3bari(i32 noundef %0) #[[ATTR:[0-9]+]]
int bar(int x) return_failure{int} {
  if (x < 0) return_failure x;
  return x + 1;
}

// CHECK-LABEL: define dso_local { i32, i1 } @_Z3fooi(i32 noundef %0)
// CHECK:         call { i32, i1 } @_Z3bari
// CHECK:         extractvalue { i32, i1 } %{{.*}}, 1
// CHECK:         br i1 %{{.*}}, label %{{[0-9]+}}, label %{{[0-9]+}}
int foo(int x) return_failure{int} {
  return bar(x);  // bare call, auto-propagates
}

// A plain (non-throws) function cannot use auto-propagation: calling a fails
// function bare must be handled explicitly with catch return_failure().
// CHECK-LABEL: define dso_local noundef i32 @_Z6bazouri(i32 noundef %0)
// CHECK:         %[[E:.*]] = alloca %struct.__herb_catch_fails, align 4
int bazour(int x) {
  auto e = catch return_failure(bar(x));
  return !e.failed ? e.value : e.error;
}

// CHECK: attributes #[[ATTR]] = { {{.*}}throws{{.*}} }
