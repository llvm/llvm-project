// This file is used from other tests.
// RUN: true

#include <thread>

#include <stdio.h>
#include <stdlib.h>

struct MyObject;
typedef MyObject *MyObjectRef;
extern "C" {
  void InitializeLibrary();
  MyObject *ObjectCreate();
  long ObjectRead(MyObject *);
  void ObjectWrite(MyObject *, long);
  void ObjectWriteAnother(MyObject *, long);
}

extern "C" void NonInstrumentedModule() {
  InitializeLibrary();
  
  MyObjectRef ref = ObjectCreate();
  std::thread t1([ref]{ ObjectWrite(ref, 42); });
  std::thread t2([ref]{ ObjectWrite(ref, 43); });
  t1.join();
  t2.join();
}
