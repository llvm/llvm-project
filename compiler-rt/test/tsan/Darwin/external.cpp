// RUN: %clangxx_tsan %p/external-lib.cpp -shared \
// RUN:                               -o %t-lib-instrumented.dylib \
// RUN:   -install_name @rpath/`basename %t-lib-instrumented.dylib`

// RUN: %clangxx_tsan %p/external-lib.cpp -shared -fno-sanitize=thread \
// RUN:                               -o %t-lib-noninstrumented.dylib \
// RUN:   -install_name @rpath/`basename %t-lib-noninstrumented.dylib`

// RUN: %clangxx_tsan %p/external-lib.cpp -shared -fno-sanitize=thread -DUSE_TSAN_CALLBACKS \
// RUN:                               -o %t-lib-noninstrumented-callbacks.dylib \
// RUN:   -install_name @rpath/`basename %t-lib-noninstrumented-callbacks.dylib`

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
  
  // TEST3: WARNING: ThreadSanitizer: race on MyLibrary::MyObject
  // TEST3: {{Modifying|read-only}} access of MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectRead}}
  // TEST3: Previous {{modifying|read-only}} access of MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectRead}}
  // TEST3: Location is MyLibrary::MyObject of size 16 at
  // TEST3: {{ObjectCreate}}
  // TEST3: SUMMARY: ThreadSanitizer: race on MyLibrary::MyObject {{.*}} in {{ObjectWrite|ObjectRead}}

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
  
  // TEST3: WARNING: ThreadSanitizer: race on MyLibrary::MyObject
  // TEST3: Modifying access of MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectWriteAnother}}
  // TEST3: Previous modifying access of MyLibrary::MyObject at
  // TEST3: {{ObjectWrite|ObjectWriteAnother}}
  // TEST3: Location is MyLibrary::MyObject of size 16 at
  // TEST3: {{ObjectCreate}}
  // TEST3: SUMMARY: ThreadSanitizer: race on MyLibrary::MyObject {{.*}} in {{ObjectWrite|ObjectWriteAnother}}

  fprintf(stderr, "WW test done\n");
  // CHECK: WW test done
}
