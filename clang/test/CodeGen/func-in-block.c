// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -emit-llvm -o - %s | FileCheck %s
// rdar: // 7860965

extern void PRINTF(const char *);
extern void B(void (^)(void));

int main(void)
{
    PRINTF(__func__);
    B(
       ^{
            PRINTF(__func__);
        }
    );
    return 0; // not reached
}

// CHECK: @__func__.__main_block_invoke = private unnamed_addr constant [18 x i8] c"main_block_invoke\00"
// CHECK: call void @PRINTF({{.*}}@__func__.__main_block_invoke
