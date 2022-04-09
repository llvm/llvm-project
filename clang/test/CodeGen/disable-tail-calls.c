// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.9.0 -emit-llvm -O1 -mdisable-tail-calls -o - < %s | FileCheck %s

typedef struct List {
  struct List *next;
  int data;
} List;

// CHECK-LABEL: define{{.*}} %struct.List* @find
List *find(List *head, int data) {
  if (!head)
    return 0;
  if (head->data == data)
    return head;
  // CHECK: call %struct.List* @find
  return find(head->next, data);
}
