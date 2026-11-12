// RUN: rm -rf %t
// RUN: clang-doc -format=html --output=%t --source-root=%S %s --
// RUN: FileCheck %s --input-file=%t/json/GlobalNamespace/_ZTV3Foo.json

class Foo {
  void getFoo();
};

int main() {
  return 0;
}

// CHECK-NOT: FileURL
// CHECK: "Filename":"DR-131697.cpp"
