// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

// Test list stuff

void *malloc(unsigned);

// Test opaque structure support.  the list type is defined later
struct list;

struct list *PassThroughList(struct list *L) {
  return L;
}


// Recursive data structure tests...

typedef struct list {
  int Data;
  struct list *Next;
} list;

list *Data;

void foo() {
  static int Foo = 0;            // Test static local variable
  Foo += 1;                      // Increment static variable

  Data = (list*)malloc(12);      // This is not a proper list allocation
}

extern list ListNode1;
list ListNode3 = { 4, 0          };
list ListNode2 = { 3, &ListNode3 };
list ListNode0 = { 1, &ListNode1 };
list ListNode1 = { 2, &ListNode2 };


list ListArray[10];

// Iterative insert fn
void InsertIntoListTail(list **L, int Data) {
  while (*L)
    L = &(*L)->Next;
  *L = (list*)malloc(sizeof(list));
  (*L)->Data = Data;
  (*L)->Next = 0;
}

// Recursive list search fn
list *FindData(list *L, int Data) {
  if (L == 0) return 0;
  if (L->Data == Data) return L;
  return FindData(L->Next, Data);
}

void foundIt(void);

// Driver fn...
void DoListStuff() {
  list *MyList = 0;
  InsertIntoListTail(&MyList, 100);
  InsertIntoListTail(&MyList, 12);
  InsertIntoListTail(&MyList, 42);
  InsertIntoListTail(&MyList, 1123);
  InsertIntoListTail(&MyList, 1213);

  if (FindData(MyList, 75)) foundIt();
  if (FindData(MyList, 42)) foundIt();
  if (FindData(MyList, 700)) foundIt();
}

