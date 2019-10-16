// Check that CFI-generated data structures are tagged with
// "#pragma clang section" attributes

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall \
// RUN:     -fno-sanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck %s

// CHECK-DAG: attributes [[ATTR:#[0-9]+]]{{.*}}bss-section{{.*}}data-section{{.*}}rodata-section
// CHECK-DAG: @.src = private unnamed_addr constant{{.*}}cfi-pragma-section.c{{.*}}[[ATTR]]
// CHECK-DAG: @{{[0-9]+}} = private unnamed_addr constant{{.*}}int (int){{.*}}[[ATTR]]
// CHECK-DAG: @{{[0-9]+}} = private unnamed_addr global{{.*}}@.src{{.*}}[[ATTR]]

typedef int (*int_arg_fn)(int);

static int int_arg1(int arg) {
    return 0;
}

static int int_arg2(int arg) {
    return 1;
}

int_arg_fn int_funcs[2] = {int_arg1, int_arg2};

#pragma clang section bss = ".bss.mycfi"
#pragma clang section data = ".data.mycfi"
#pragma clang section rodata = ".rodata.mycfi"

int main(int argc, const char *argv[]) {

    int idx = argv[1][0] - '0';
    return int_funcs[argc](idx);
}
