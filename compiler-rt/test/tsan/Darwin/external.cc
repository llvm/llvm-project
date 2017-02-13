// RUN: %clangxx_tsan %s -o %t-lib-instrumented.dylib              -shared -DSHARED_LIB
// RUN: %clangxx_tsan %s -o %t-lib-noninstrumented.dylib           -shared -DSHARED_LIB -fno-sanitize=thread
// RUN: %clangxx_tsan %s -o %t-lib-noninstrumented-callbacks.dylib -shared -DSHARED_LIB -fno-sanitize=thread -DUSE_TSAN_CALLBACKS
// RUN: %clangxx_tsan %s %t-lib-instrumented.dylib -o %t-lib-instrumented
// RUN: %clangxx_tsan %s %t-lib-noninstrumented.dylib -o %t-lib-noninstrumented
// RUN: %clangxx_tsan %s %t-lib-noninstrumented-callbacks.dylib -o %t-lib-noninstrumented-callbacks

// RUN: %deflake %run %t-lib-instrumented              2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=TEST1
// RUN:          %run %t-lib-noninstrumented           2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=TEST2
// RUN: %deflake %run %t-lib-noninstrumented-callbacks 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=TEST3

#include <thread>

#include <dlfcn.h>
#include <pthread.h>
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

#if defined(SHARED_LIB)

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

#else  // defined(SHARED_LIB)

int main(int argc, char *argv[]) {
  InitializeLibrary();
  
  {
    MyObjectRef ref = ObjectCreate();
    std::thread t1([ref]{ ObjectRead(ref); });
    std::thread t2([ref]{ ObjectRead(ref); });
    t1.join();
    t2.join();
  }
  
  // CHECK-NOT: WARNING: ThreadSanitizer
  
  fprintf(stderr, "RR test done\n");
  // CHECK: RR test done

  {
    MyObjectRef ref = ObjectCreate();
    std::thread t1([ref]{ ObjectRead(ref); });
    std::thread t2([ref]{ ObjectWrite(ref, 66); });
    t1.join();
    t2.join();
  }
  
  // TEST1: WARNING: ThreadSanitizer: data race
  // TEST1: {{Write|Read}} of size 8 at
  // TEST1: Previous {{write|read}} of size 8 at
  // TEST1: Location is heap block of size 16 at
  
  // TEST2-NOT: WARNING: ThreadSanitizer
  
  // TEST3: WARNING: ThreadSanitizer: race on a library object
  // TEST3: {{Mutating|read-only}} access of object MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectRead}}
  // TEST3: Previous {{mutating|read-only}} access of object MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectRead}}
  // TEST3: Location is MyLibrary::MyObject object of size 16 at
  // TEST3: {{ObjectCreate}}

  fprintf(stderr, "RW test done\n");
  // CHECK: RW test done

  {
    MyObjectRef ref = ObjectCreate();
    std::thread t1([ref]{ ObjectWrite(ref, 76); });
    std::thread t2([ref]{ ObjectWriteAnother(ref, 77); });
    t1.join();
    t2.join();
  }
  
  // TEST1-NOT: WARNING: ThreadSanitizer: data race
  
  // TEST2-NOT: WARNING: ThreadSanitizer
  
  // TEST3: WARNING: ThreadSanitizer: race on a library object
  // TEST3: Mutating access of object MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectWriteAnother}}
  // TEST3: Previous mutating access of object MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectWriteAnother}}
  // TEST3: Location is MyLibrary::MyObject object of size 16 at
  // TEST3: {{ObjectCreate}}

  fprintf(stderr, "WW test done\n");
  // CHECK: WW test done
}

#endif  // defined(SHARED_LIB)
