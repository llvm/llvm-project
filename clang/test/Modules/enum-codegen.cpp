// RUN: rm -rf %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fmodules-cache-path=%t %s -emit-llvm -o - | FileCheck %s

// CHECK: @{{.*var.*}} = {{.*}} %union.union_type { i8 1 },

#pragma clang module build bar
module bar {
  header "bar.h" { size 40 mtime 0 }
  export *
}
#pragma clang module contents
#pragma clang module begin bar
union union_type {
  char h{1};
};
#pragma clang module end
#pragma clang module endbuild
#pragma clang module build foo
module foo {
  header "foo.h" { size 97 mtime 0 }
  export *
}
#pragma clang module contents
#pragma clang module begin foo
union union_type {
  char h{1};
};
#pragma clang module import bar
template<typename T>
union_type var;
#pragma clang module end
#pragma clang module endbuild
#pragma clang module import foo
int main() {
  (void)&var<int>;
}
