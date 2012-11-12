// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: store i32 59, i32* %size
// CHECK: store i32 65, i32* %size
template<typename T>
class TemplateClass {
public:
  void templateClassFunction() {
    int size = sizeof(__PRETTY_FUNCTION__);
  }
};

// CHECK: store i32 35, i32* %size
// CHECK: store i32 38, i32* %size
template<typename T>
void functionTemplate(T t) {
  int size = sizeof(__PRETTY_FUNCTION__);
}

int main() {
  TemplateClass<int> t1;
  t1.templateClassFunction();
  TemplateClass<double> t2;
  t2.templateClassFunction();

  functionTemplate<int>(0);
  functionTemplate(0.0);

  return 0;
}
