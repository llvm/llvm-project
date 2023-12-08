// RUN: %clangxx_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import <memory>

struct InnerStruct {
  ~InnerStruct() {
    fprintf(stderr, "~InnerStruct\n");
  }
};

struct MyStruct {
  std::shared_ptr<InnerStruct> inner_object;
  ~MyStruct() {
    fprintf(stderr, "~MyStruct\n");
  }
};

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  {
    std::shared_ptr<MyStruct> shared(new MyStruct());
    shared->inner_object = std::shared_ptr<InnerStruct>(new InnerStruct());
  }

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: ~MyStruct
// CHECK: ~InnerStruct
// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
