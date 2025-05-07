

// RUN: %clang_cc1 -O0 -fbounds-safety -fsanitize=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=RECOVER %s
// RUN: %clang_cc1 -O2 -fbounds-safety -fsanitize=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=RECOVER %s
// RUN: %clang_cc1 -O0 -fbounds-safety -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=TRAP %s
// RUN: %clang_cc1 -O2 -fbounds-safety -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=TRAP %s
// RUN: %clang_cc1 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsanitize=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=RECOVER %s
// RUN: %clang_cc1 -O2 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsanitize=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=RECOVER %s
// RUN: %clang_cc1 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=TRAP %s
// RUN: %clang_cc1 -O2 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm %s -o - | FileCheck --check-prefix=TRAP %s

struct foo {
    unsigned bar;
};

struct foo arr[10];

int v;

int main() {
    struct foo *ptr = &arr[v + 1];
// RECOVER: call void @__ubsan_handle_out_of_bounds

// TRAP: br {{.*}} label %[[LABEL_TRAP:[a-z0-9]+]], !prof ![[PROFILE_METADATA:[0-9]+]], !nosanitize
// TRAP: [[LABEL_TRAP]]:
// TRAP-NEXT:  call void @llvm.ubsantrap(i8 {{.*}}) #{{.*}}, !nosanitize

}
