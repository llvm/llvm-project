// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -fblocks -triple %itanium_abi_triple %s -o - | FileCheck %s

// CHECK: !DISubprogram(name: "___Z1fU13block_pointerFviE_block_invoke", linkageName: "___Z1fU13block_pointerFviE_block_invoke"
void g(void (^call)(int));

void f(void (^callback)(int)) {
  g(^(int x) {
    callback(x);
  });
}

void h() {
  f(^(int x){
  });
}
