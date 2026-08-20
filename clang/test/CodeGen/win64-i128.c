// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang --target=x86_64-windows-msvc -mno-sse -S -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefixes=CHECK,NOSSE
// RUN: %clang_cc1 -triple aarch64-windows-msvc -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefixes=CHECK,ARM,ARM64
// RUN: %clang_cc1 -triple arm64ec-windows-msvc -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefixes=CHECK,ARM,ARM64EC

typedef int int128_t __attribute__((mode(TI)));

int128_t foo(void) { return 0; }

// X64-LABEL: define dso_local void @foo(
// X64-SAME: sret(i128) align 16
// ARM: define dso_local i128 @foo()

int128_t bar(int128_t a, int128_t b) { return a * b; }

// X64-LABEL: define dso_local void @bar(
// X64-SAME: sret(i128) align 16
// X64-SAME: ptr nofree noundef align 16 dead_on_return
// X64-SAME: ptr nofree noundef align 16 dead_on_return
// NOSSE-LABEL: define dso_local void @bar(
// NOSSE-SAME: sret(i128) align 16
// NOSSE-SAME: ptr nofree noundef align 16 dead_on_return
// NOSSE-SAME: ptr nofree noundef align 16 dead_on_return
// ARM: define dso_local i128 @bar(i128 noundef %a, i128 noundef %b)

#if defined(__x86_64__) && !defined(__arm64ec__)
int128_t __attribute__((vectorcall)) vectorcall_foo(void) { return 0; }

// X64-LABEL: define dso_local x86_vectorcallcc void @"\01vectorcall_foo@@0"(
// X64-SAME: sret(i128) align 16

int128_t __attribute__((vectorcall)) vectorcall_bar(int128_t a, int128_t b) {
  return a * b;
}

// X64-LABEL: define dso_local x86_vectorcallcc void @"\01vectorcall_bar@@32"(
// X64-SAME: sret(i128) align 16
// X64-SAME: ptr nofree noundef align 16 dead_on_return
// X64-SAME: ptr nofree noundef align 16 dead_on_return
#endif

void vararg(int a, ...) {
  // CHECK: define{{.*}} void @vararg
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  int128_t i = __builtin_va_arg(ap, int128_t);

  // __int128 is passed indirectly, so there is a double load.
  //
  // X64: load ptr, ptr
  // X64: load i128, ptr

  // Explicitly check that the read is properly aligned.
  //
  // ARM64: %argp.cur = load ptr, ptr %ap
  // ARM64: %argp.cur.aligned = call ptr @llvm.ptrmask.p0.i64(ptr %{{.*}}, i64 -16)
  // ARM64: load i128, ptr %argp.cur.aligned, align 16

  // On ARM64EC __int128 is passed indirectly, so there is a double load.
  //
  // ARM64EC: %argp.cur = load ptr, ptr %ap
  // ARM64EC: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 8
  // ARM64EC: [[P:%.*]] = load ptr, ptr %argp.cur
  // ARM64EC: load i128, ptr [[P]], align 16
  __builtin_va_end(ap);
}

struct Align16 {
  char x[16];
} __attribute__((aligned(16)));

void vararg_struct(int a, ...) {
  // CHECK: define{{.*}} void @vararg_struct
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  struct Align16 i = __builtin_va_arg(ap, struct Align16);

  // X64,ARM64EC: %argp.cur = load ptr, ptr %ap
  // X64,ARM64EC: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 8
  // X64,ARM64EC: store ptr %argp.next, ptr %ap
  // X64,ARM64EC: [[P:%.*]] = load ptr, ptr %argp.cur
  // X64,ARM64EC: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %i, ptr align 16 [[P]], i64 16, i1 false)

  // ARM64: %argp.cur = load ptr, ptr %ap
  // ARM64: [[ADD:%.*]] = getelementptr inbounds i8, ptr %argp.cur, i32 15
  // ARM64: %argp.cur.aligned = call ptr @llvm.ptrmask.p0.i64(ptr [[ADD]], i64 -16)
  // ARM64: %argp.next = getelementptr inbounds i8, ptr %argp.cur.aligned, i64 16
  // ARM64: store ptr %argp.next, ptr %ap
  // ARM64: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %i, ptr align 16 %argp.cur.aligned, i64 16, i1 false)

  __builtin_va_end(ap);
}
