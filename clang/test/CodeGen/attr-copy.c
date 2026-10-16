// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --implicit-check-not='@allocate(' --implicit-check-not='@stop('

void *allocate(unsigned long) __attribute__((malloc, alloc_size(1), nothrow));
void *copied(unsigned long) __attribute__((copy(allocate)));

void *use(unsigned long n) {
  return copied(n);
}

int aligned_source __attribute__((aligned(64)));
// CHECK-DAG: @aligned_copy = {{.*}}global i32 0, align 64
int aligned_copy __attribute__((copy(aligned_source)));

// CHECK-DAG: @strong_alias = {{.*}}alias ptr (i64), ptr @target
void *target(unsigned long) __attribute__((malloc, alloc_size(1), nothrow));
void *target(unsigned long n) { return (void *)0; }
extern __typeof__(target) strong_alias
    __attribute__((alias("target"), copy(target)));

// CHECK: declare noalias ptr @copied(i64 noundef) #[[ALLOC:[0-9]+]]

void stop(void) __attribute__((noreturn));
void copied_stop(void) __attribute__((copy(stop)));
void use_stop(void) {
  copied_stop();
}
// CHECK: declare void @copied_stop() #[[STOP:[0-9]+]]

void *use_alias(unsigned long n) {
  return strong_alias(n);
}
// CHECK: call noalias ptr @strong_alias(i64 noundef
// CHECK: attributes #[[ALLOC]] = { nounwind allocsize(0)
// CHECK: attributes #[[STOP]] = { noreturn
