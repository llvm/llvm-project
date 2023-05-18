// RUN: %clang_cc1 -emit-llvm -fobjc-exceptions -triple x86_64-unknown-linux -o - %s | FileCheck %s

// Test that the nomerge attribute is applied to function calls
// in @try, @catch and @finally
void opaque(void);
void opaque2(void);
void opaque3(void);

int main(int argc, const char * argv[]) {
  __attribute__((nomerge)) @try {
    opaque();
  } @catch(...) {
    opaque2();
  } @finally {
    opaque3();
  }

  return 0;
}

// CHECK: call void @opaque() #[[ATTR0:[0-9]+]]
// CHECK-DAG: call void @opaque2() #[[ATTR0]]
// CHECK-DAG: call void @opaque3() #[[ATTR0]]
// CHECK-DAG: attributes #[[ATTR0]] = {{{.*}}nomerge{{.*}}}
