// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm < %s | FileCheck %s

_Static_assert(sizeof(void *) == 8, "sizeof(void *) has unexpected value.  Expected 8.");

int foo(void) {
  // CHECK: define dso_local i32 @foo
  // CHECK: %a = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  int (*__ptr32 a)(int);
  return sizeof(a);
}

int bar(void) {
  // CHECK: define dso_local i32 @bar
  // CHECK: %p = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  int *__ptr32 p;
  return sizeof(p);
}


int baz(void) {
  // CHECK: define dso_local i32 @baz
  // CHECK: %p = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  typedef int *__ptr32 IP32_PTR;

  IP32_PTR p;
  return sizeof(p);
}

int fugu(void) {
  // CHECK: define dso_local i32 @fugu
  // CHECK: %p = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  typedef int *int_star;

  int_star __ptr32 p;
  return sizeof(p);
}

typedef __SIZE_TYPE__ size_t;
size_t strlen(const char *);

size_t test_calling_strlen_with_32_bit_pointer ( char *__ptr32 s ) {
  // CHECK-LABEL: define dso_local i64 @test_calling_strlen_with_32_bit_pointer(ptr addrspace(270) noundef %s)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %s.addr = alloca ptr addrspace(270), align 4
  // CHECK-NEXT:   store ptr addrspace(270) %s, ptr %s.addr, align 4
  // CHECK-NEXT:   %0 = load ptr addrspace(270), ptr %s.addr, align 4
  // CHECK-NEXT:   %1 = addrspacecast ptr addrspace(270) %0 to ptr
  // CHECK-NEXT:   %call = call i64 @strlen(ptr  noundef %1)
  // CHECK-NEXT:   ret i64 %call
   return strlen ( s );
}

// CHECK-LABEL: declare dso_local i64 @strlen(ptr noundef)

size_t test_calling_strlen_with_64_bit_pointer ( char *s ) {
  // CHECK-LABEL: define dso_local i64 @test_calling_strlen_with_64_bit_pointer(ptr noundef %s)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %s.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %s, ptr %s.addr, align 8
  // CHECK-NEXT:   %0 = load ptr, ptr %s.addr, align 8
  // CHECK-NEXT:   %call = call i64 @strlen(ptr noundef %0)
  // CHECK-NEXT:   ret i64 %call
  return strlen ( s );
}
