// RUN: %clang --sycl -c %s  -o %t.ll -Xclang -fsycl-int-header=%t.hpp -emit-llvm -S
// RUN: FileCheck < %t.ll %s --check-prefix=CHECK

// CHECK-NOT: declare dso_local spir_func void {{.+}}test{{.+}}printer{{.+}}
class test {
public:
  virtual void printer();
};

void test::printer() {}
