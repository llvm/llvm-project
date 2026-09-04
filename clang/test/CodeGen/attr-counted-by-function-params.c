// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-late-parse-attributes -DLATE_PARSING -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LATE

// Test that counted_by on a function parameter is consumed entirely by Sema:
// CodeGen ignores it, and late parsing does not change what is emitted.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

// CHECK-LABEL: define dso_local i32 @sum(
// CHECK-SAME:    i32 noundef %count, ptr noundef %buf)
// CHECK-NOT:     counted_by
// CHECK:         getelementptr inbounds i32, ptr
int sum(int count, int *__counted_by(count) buf) {
  int t = 0;
  for (int i = 0; i < count; ++i)
    t += buf[i];
  return t;
}

// CHECK-LABEL: define dso_local i64 @bdos(
// CHECK-NOT:     counted_by.load
// CHECK:         call i64 @llvm.objectsize.i64.p0(
unsigned long bdos(int count, int *__counted_by(count) buf) {
  return __builtin_dynamic_object_size(buf, 0);
}

// CHECK-LABEL: define dso_local void @family(
// CHECK-SAME:    i32 noundef %n, ptr noundef %a, ptr noundef %b, ptr noundef %c)
// CHECK-NOT:     counted_by
//
// Verify: the _or_null and byte-count spellings behave the same way
void family(int n, int *__counted_by(n) a, int *__counted_by_or_null(n) b,
            void *__sized_by(n) c) {
  (void)a;
  (void)b;
  (void)c;
}

// CHECK-LABEL: define dso_local void @takes_cb(
// CHECK-SAME:    i32 noundef %n, ptr noundef %cb, ptr noundef %p)
// CHECK:         call void %{{.*}}(i32 noundef %{{.*}}, ptr noundef %{{.*}})
//
// Verify: an attribute inside a function-pointer parameter's own prototype
// does not change the callee's signature
void takes_cb(int n, void (*cb)(int m, int *__counted_by(m) q),
              int *__counted_by(n) p) {
  cb(n, p);
}

void sum_redecl(int count, int *__counted_by(count) buf);
void sum_redecl(int count, int *buf);

// CHECK-LABEL: define dso_local i32 @call_site(
// CHECK:         call i32 @sum(i32 noundef 4, ptr noundef %{{.*}})
//
// Verify: declaring with the attribute and defining without it does not split
// the symbol
int call_site(void) {
  int a[4];
  return sum(4, a);
}

#ifdef LATE_PARSING
// LATE-LABEL: define dso_local i32 @fwd_ref(
// LATE-SAME:    ptr noundef %buf, i32 noundef %count)
// LATE-NOT:     counted_by
// LATE:         getelementptr inbounds i32, ptr
//
// Verify: a count declared after the annotated parameter emits identically
int fwd_ref(int *__counted_by(count) buf, int count) { return buf[count - 1]; }
#endif
