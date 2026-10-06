// RUN: %clang_cc1 -triple x86_64-linux -O1 -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=OLD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -new-struct-path-tbaa \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=NEW
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes \
// RUN:   -new-struct-path-tbaa -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR

struct Pair {
  int x, y;
};
union U {
  int i;
  struct Pair pair;
};
struct Outer {
  union U u;
  int sibling;
} *outer;
union U *dest, *src;

// OLD-LABEL: define{{.*}} i32 @distinct_object(
// OLD: load i32, ptr
// OLD: ret i32
// NEW-LABEL: define{{.*}} i32 @distinct_object(
// NEW-NOT: load i32, ptr
// NEW: ret i32 123
int distinct_object(void) {
  outer->sibling = 123;
  *dest = *src;
  return outer->sibling;
}

// A pointer to a union member must still alias a whole-union store.
// NEW-LABEL: define{{.*}} i32 @escaped_member(
// NEW: load i32, ptr
// NEW: ret i32
int escaped_member(void) {
  int *p = &dest->i;
  *p = 123;
  *dest = *src;
  return *p;
}

// This also applies to pointers to nested members.
// NEW-LABEL: define{{.*}} i32 @escaped_nested_member(
// NEW: load i32, ptr
// NEW: ret i32
int escaped_nested_member(void) {
  int *p = &dest->pair.y;
  *p = 123;
  *dest = *src;
  return *p;
}

// IR: call void @llvm.memcpy{{.*}}, !tbaa [[TAG_U:![0-9]+]]
// IR-DAG: [[CHAR:![0-9]+]] = !{!{{[0-9]+}}, i64 1, !"omnipotent char"}
// IR-DAG: [[INT:![0-9]+]] = !{[[CHAR]], i64 4, !"int"}
// IR-DAG: [[PAIR:![0-9]+]] = !{[[CHAR]], i64 8, !"Pair", [[INT]], i64 0, i64 4, [[INT]], i64 4, i64 4}
// IR-DAG: [[UNION:![0-9]+]] = !{[[CHAR]], i64 8, !"U", [[INT]], i64 0, i64 4, [[PAIR]], i64 0, i64 8}
// IR-DAG: [[TAG_U]] = !{[[UNION]], [[UNION]], i64 0, i64 8}
