// RUN: %clang_cc1 -emit-llvm -triple=x86_64-pc-win32 -fms-compatibility %s -o - | FileCheck %s

class Test1 {
private:
  int x_;

public:
  Test1(int x) : x_(x) {}
  __declspec(property(get = get_x)) int X;
  int get_x() const { return x_; }
  static Test1 *GetTest1() { return new Test1(10); }
};

// CHECK-LABEL: main
int main(int argc, char **argv) {
  // CHECK: [[CALL:%.+]] = call %class.Test1* @"\01?GetTest1@Test1@@SAPEAV1@XZ"()
  // CHECK-NEXT: call i32 @"\01?get_x@Test1@@QEBAHXZ"(%class.Test1* [[CALL]])
  return Test1::GetTest1()->X;
}
