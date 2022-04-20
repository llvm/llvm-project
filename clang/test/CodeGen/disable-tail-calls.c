// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -opaque-pointers -emit-llvm -O2 -fno-optimize-sibling-calls -o - < %s | FileCheck %s

typedef struct List {
  struct List *next;
  int data;
} List;

// CHECK-LABEL: define{{.*}} ptr @find(
List *find(List *head, int data) {
  if (!head)
    return 0;
  if (head->data == data)
    return head;
  // CHECK: call ptr @find(
  return find(head->next, data);
}
