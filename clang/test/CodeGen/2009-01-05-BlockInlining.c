// RUN: %clang_cc1 %s -triple x86_64-linux -emit-llvm -fblocks -o - | FileCheck %s

// These will be inlined by the optimizers provided the block descriptors
// and block literals are internal constants.
// CHECK: @__block_descriptor_tmp = internal constant
// CHECK: @__block_literal_global = internal constant
// CHECK: @__block_descriptor_tmp.2 = internal constant
// CHECK: @__block_literal_global.3 = internal constant
static int fun(int x) {
	return x+1;
}

static int block(int x) {
	return (^(int x){return x+1;})(x);
}

extern int printf(const char *, ...);
static void print(int result) {
    printf("%d\n", result);
}

int main (int argc, const char * argv[]) {
    int	x = argc-1;
    print(fun(x));
    print(block(x));
    int	(^block_inline)(int) = ^(int x){return x+1;};
    print(block_inline(x));
    return 0;
}
