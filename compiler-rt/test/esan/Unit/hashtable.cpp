// RUN: %clangxx_unit -esan-instrument-loads-and-stores=0 -O0 %s -o %t 2>&1
// RUN: %env_esan_opts="record_snapshots=0" %run %t 2>&1 | FileCheck %s

#include "esan/esan_hashtable.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class MyData {
 public:
  MyData(const char *Str) : RefCount(0) { Buf = strdup(Str); }
  ~MyData() {
    fprintf(stderr, "  Destructor: %s.\n", Buf);
    free(Buf);
  }
  bool operator==(MyData &Cmp) { return strcmp(Buf, Cmp.Buf) == 0; }
  operator size_t() const {
    size_t Res = 0;
    for (int i = 0; i < strlen(Buf); ++i)
      Res ^= Buf[i];
    return Res;
  }
  char *Buf;
  int RefCount;
};

// We use a smart pointer wrapper to free the payload on hashtable removal.
struct MyDataPayload {
  MyDataPayload() : Data(nullptr) {}
  explicit MyDataPayload(MyData *Data) : Data(Data) { ++Data->RefCount; }
  ~MyDataPayload() {
    if (Data && --Data->RefCount == 0) {
      fprintf(stderr, "Deleting %s.\n", Data->Buf);
      delete Data;
    }
  }
  MyDataPayload(const MyDataPayload &Copy) {
    Data = Copy.Data;
    ++Data->RefCount;
  }
  MyDataPayload & operator=(const MyDataPayload &Copy) {
    if (this != &Copy) {
      this->~MyDataPayload();
      Data = Copy.Data;
      ++Data->RefCount;
    }
    return *this;
  }
  bool operator==(MyDataPayload &Cmp) { return *Data == *Cmp.Data; }
  operator size_t() const { return (size_t)*Data; }
  MyData *Data;
};

int main()
{
  __esan::HashTable<int, int> IntTable;
  assert(IntTable.size() == 0);

  // Test iteration on an empty table.
  int Count = 0;
  for (auto Iter = IntTable.begin(); Iter != IntTable.end();
       ++Iter, ++Count) {
    // Empty.
  }
  assert(Count == 0);

  bool Added = IntTable.add(4, 42);
  assert(Added);
  assert(!IntTable.add(4, 42));
  assert(IntTable.size() == 1);
  int Value;
  bool Found = IntTable.lookup(4, Value);
  assert(Found && Value == 42);

  // Test iterator.
  IntTable.lock();
  for (auto Iter = IntTable.begin(); Iter != IntTable.end();
       ++Iter, ++Count) {
    assert((*Iter).Key == 4);
    assert((*Iter).Data == 42);
  }
  IntTable.unlock();
  assert(Count == 1);
  assert(Count == IntTable.size());
  assert(!IntTable.remove(5));
  assert(IntTable.remove(4));

  // Test a more complex payload.
  __esan::HashTable<int, MyDataPayload> DataTable(4);
  MyDataPayload NewData(new MyData("mystring"));
  Added = DataTable.add(4, NewData);
  assert(Added);
  MyDataPayload FoundData;
  Found = DataTable.lookup(4, FoundData);
  assert(Found && strcmp(FoundData.Data->Buf, "mystring") == 0);
  assert(!DataTable.remove(5));
  assert(DataTable.remove(4));
  // Test resize.
  for (int i = 0; i < 4; ++i) {
    MyDataPayload MoreData(new MyData("delete-at-end"));
    Added = DataTable.add(i+1, MoreData);
    assert(Added);
    assert(!DataTable.add(i+1, MoreData));
  }
  for (int i = 0; i < 4; ++i) {
    Found = DataTable.lookup(i+1, FoundData);
    assert(Found && strcmp(FoundData.Data->Buf, "delete-at-end") == 0);
  }
  DataTable.lock();
  Count = 0;
  for (auto Iter = DataTable.begin(); Iter != DataTable.end();
       ++Iter, ++Count) {
    int Key = (*Iter).Key;
    FoundData = (*Iter).Data;
    assert(Key >= 1 && Key <= 4);
    assert(strcmp(FoundData.Data->Buf, "delete-at-end") == 0);
  }
  DataTable.unlock();
  assert(Count == 4);
  assert(Count == DataTable.size());

  // Ensure the iterator supports a range-based for loop.
  DataTable.lock();
  Count = 0;
  for (auto Pair : DataTable) {
    assert(Pair.Key >= 1 && Pair.Key <= 4);
    assert(strcmp(Pair.Data.Data->Buf, "delete-at-end") == 0);
    ++Count;
  }
  DataTable.unlock();
  assert(Count == 4);
  assert(Count == DataTable.size());

  // Test payload freeing via smart pointer wrapper.
  __esan::HashTable<MyDataPayload, MyDataPayload, true> DataKeyTable;
  MyDataPayload DataA(new MyData("string AB"));
  DataKeyTable.lock();
  Added = DataKeyTable.add(DataA, DataA);
  assert(Added);
  Found = DataKeyTable.lookup(DataA, FoundData);
  assert(Found && strcmp(FoundData.Data->Buf, "string AB") == 0);
  MyDataPayload DataB(new MyData("string AB"));
  Added = DataKeyTable.add(DataB, DataB);
  assert(!Added);
  DataKeyTable.remove(DataB); // Should free the DataA payload.
  DataKeyTable.unlock();

  // Test custom functors.
  struct CustomHash {
    size_t operator()(int Key) const { return Key % 4; }
  };
  struct CustomEqual {
    bool operator()(int Key1, int Key2) const { return Key1 %4 == Key2 % 4; }
  };
  __esan::HashTable<int, int, false, CustomHash, CustomEqual> ModTable;
  Added = ModTable.add(2, 42);
  assert(Added);
  Added = ModTable.add(6, 42);
  assert(!Added);

  fprintf(stderr, "All checks passed.\n");
  return 0;
}
// CHECK:      Deleting mystring.
// CHECK-NEXT:   Destructor: mystring.
// CHECK-NEXT: All checks passed.
// CHECK-NEXT: Deleting string AB.
// CHECK-NEXT:   Destructor: string AB.
// CHECK-NEXT: Deleting string AB.
// CHECK-NEXT:   Destructor: string AB.
// CHECK-NEXT: Deleting delete-at-end.
// CHECK-NEXT:   Destructor: delete-at-end.
// CHECK-NEXT: Deleting delete-at-end.
// CHECK-NEXT:   Destructor: delete-at-end.
// CHECK-NEXT: Deleting delete-at-end.
// CHECK-NEXT:   Destructor: delete-at-end.
// CHECK-NEXT: Deleting delete-at-end.
// CHECK-NEXT:   Destructor: delete-at-end.
