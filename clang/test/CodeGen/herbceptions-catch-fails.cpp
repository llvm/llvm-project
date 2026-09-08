// RUN: %clang -fherbceptions -fno-exceptions -S -emit-llvm -o - %s | FileCheck %s

// Herbception `catch return_failure(expr)`: the throws call returns {T, i1}, and the
// expression builds the N2289 aggregate
//   struct { union { T value; E error; }; bool failed; }
// with .failed = discriminant and .value/.error sourced from the payload slot.

// CHECK: %struct.__herb_catch_fails = type { %union.anon, i8 }

// A return_failure{int} function returns {i32, i1} with the 'throws' attribute.
// CHECK: define dso_local { i32, i1 } @_Z3bari(i32 noundef %0) #[[ATTR:[0-9]+]]
int bar(int x) return_failure{int} {
  if (x < 0) return_failure x;
  return x + 1;
}

// catch return_failure(bar(x)) extracts the discriminant and builds the aggregate.
// CHECK-LABEL: define dso_local noundef i32 @_Z3fooi(i32 noundef %0)
// CHECK:         %{{.*}} = call { i32, i1 } @_Z3bari
// CHECK:         %{{.*}} = extractvalue { i32, i1 } %{{.*}}, 0
// CHECK:         %{{.*}} = extractvalue { i32, i1 } %{{.*}}, 1
// CHECK:         getelementptr inbounds nuw %struct.__herb_catch_fails, ptr %{{.*}}, i32 0, i32 1
// CHECK:         %{{.*}} = zext i1 %{{.*}} to i8
// CHECK:         store i8 %{{.*}}, ptr %{{.*}}, align 4
// CHECK:         getelementptr inbounds nuw %struct.__herb_catch_fails, ptr %{{.*}}, i32 0, i32 0
// CHECK:         store i32 %{{.*}}, ptr %{{.*}}, align 4
int foo(int x) {
  auto e = catch return_failure(bar(x));
  return !e.failed ? e.value * 2 : e.error;
}

// CHECK: attributes #[[ATTR]] = { {{.*}}throws{{.*}} }
