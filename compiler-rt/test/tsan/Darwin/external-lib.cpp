// This file is used from other tests.
// RUN: true

#include <dlfcn.h>
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

struct MyObject {
  long _val;
  long _another;
};

#if defined(USE_TSAN_CALLBACKS)
static void *tag;
void *(*callback_register_tag)(const char *object_type);
void *(*callback_assign_tag)(void *addr, void *tag);
void (*callback_read)(void *addr, void *caller_pc, void *tag);
void (*callback_write)(void *addr, void *caller_pc, void *tag);
#endif

void InitializeLibrary() {
#if defined(USE_TSAN_CALLBACKS)
  callback_register_tag = (decltype(callback_register_tag))dlsym(RTLD_DEFAULT, "__tsan_external_register_tag");
  callback_assign_tag = (decltype(callback_assign_tag))dlsym(RTLD_DEFAULT, "__tsan_external_assign_tag");
  callback_read = (decltype(callback_read))dlsym(RTLD_DEFAULT, "__tsan_external_read");
  callback_write = (decltype(callback_write))dlsym(RTLD_DEFAULT, "__tsan_external_write");
  tag = callback_register_tag("MyLibrary::MyObject");
#endif
}

MyObject *ObjectCreate() {
  MyObject *ref = (MyObject *)malloc(sizeof(MyObject));
#if defined(USE_TSAN_CALLBACKS)
  callback_assign_tag(ref, tag);
#endif
  return ref;
}

long ObjectRead(MyObject *ref) {
#if defined(USE_TSAN_CALLBACKS)
  callback_read(ref, __builtin_return_address(0), tag);
#endif
  return ref->_val;
}

void ObjectWrite(MyObject *ref, long val) {
#if defined(USE_TSAN_CALLBACKS)
  callback_write(ref, __builtin_return_address(0), tag);
#endif
  ref->_val = val;
}

void ObjectWriteAnother(MyObject *ref, long val) {
#if defined(USE_TSAN_CALLBACKS)
  callback_write(ref, __builtin_return_address(0), tag);
#endif
  ref->_another = val;
}
