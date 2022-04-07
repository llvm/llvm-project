// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s

// C guarantees that brace-init with fewer initializers than members in the
// aggregate will initialize the rest of the aggregate as-if it were static
// initialization. In turn static initialization guarantees that padding is
// initialized to zero bits.

// CHECK: @__const.partial_init.s = private unnamed_addr constant { i8, [7 x i8], i64 } { i8 42, [7 x i8] zeroinitializer, i64 0 }, align 8

// Technically, we could initialize this padding to non-zero because all of the
// struct's members have initializers.

// CHECK: @__const.init_all.s = private unnamed_addr constant { i8, [7 x i8], i64 } { i8 42, [7 x i8] zeroinitializer, i64 -2401053089374216531 }, align 8

struct S {
  char c;
  long long l;
};

void use(struct S*);

// CHECK-LABEL: @empty_braces(
// CHECK:       %s = alloca
// CHECK-NEXT:  %[[B:[0-9+]]] = bitcast %struct.S* %s to i8*
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 8 %[[B]], i8 0,
// CHECK-NEXT:  call void @use(%struct.S* noundef %s)
void empty_braces(void) {
  struct S s = {};
  return use(&s);
}

// CHECK-LABEL: @partial_init(
// CHECK:       %s = alloca
// CHECK-NEXT:  %[[B:[0-9+]]] = bitcast %struct.S* %s to i8*
// CHECK-NEXT:  call void @llvm.memcpy{{.*}}(i8* align 8 %[[B]], {{.*}}@__const.partial_init.s
// CHECK-NEXT:  call void @use(%struct.S* noundef %s)
void partial_init(void) {
  struct S s = { .c = 42 };
  return use(&s);
}

// CHECK-LABEL: @init_all(
// CHECK:       %s = alloca
// CHECK-NEXT:  %[[B:[0-9+]]] = bitcast %struct.S* %s to i8*
// CHECK-NEXT:  call void @llvm.memcpy{{.*}}(i8* align 8 %[[B]], {{.*}}@__const.init_all.s
// CHECK-NEXT:  call void @use(%struct.S* noundef %s)
void init_all(void) {
  struct S s = { .c = 42, .l = 0xdeadbeefc0fedead };
  return use(&s);
}
