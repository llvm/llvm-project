// RUN: %clang_cc1 -no-opaque-pointers -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -Werror -triple i686-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -no-opaque-pointers -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -Werror -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=X64

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

#pragma intrinsic(__cpuid)

void test__cpuid(int cpuInfo[4], int function_id) {
  __cpuid(cpuInfo, function_id);
}
// X86-LABEL: define {{.*}} @test__cpuid(i32* noundef %{{.*}}, i32 noundef %{{.*}})
// X86-DAG: [[ASMRESULTS:%[0-9]+]] = call { i32, i32, i32, i32 } asm "cpuid", "={ax},={bx},={cx},={dx},{ax},{cx}"
// X86-DAG: [[ADDRPTR0:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 0
// X86-DAG: [[ADDRPTR1:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 1
// X86-DAG: [[ADDRPTR2:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 2
// X86-DAG: [[ADDRPTR3:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 3
// X86-DAG: [[RESULT0:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// X86-DAG: [[RESULT1:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// X86-DAG: [[RESULT2:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// X86-DAG: [[RESULT3:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// X86-DAG: store i32 [[RESULT0]], i32* [[ADDRPTR0]], align 4
// X86-DAG: store i32 [[RESULT1]], i32* [[ADDRPTR1]], align 4
// X86-DAG: store i32 [[RESULT2]], i32* [[ADDRPTR2]], align 4
// X86-DAG: store i32 [[RESULT3]], i32* [[ADDRPTR3]], align 4

// X64-LABEL: define {{.*}} @test__cpuid(i32* noundef %{{.*}}, i32 noundef %{{.*}})
// X64-DAG: [[ASMRESULTS:%[0-9]+]] = call { i32, i32, i32, i32 } asm "xchgq %rbx, ${1:q}\0Acpuid\0Axchgq %rbx, ${1:q}", "={ax},=r,={cx},={dx},0,2"
// X64-DAG: [[ADDRPTR0:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 0
// X64-DAG: [[ADDRPTR1:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 1
// X64-DAG: [[ADDRPTR2:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 2
// X64-DAG: [[ADDRPTR3:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 3
// X64-DAG: [[RESULT0:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// X64-DAG: [[RESULT1:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// X64-DAG: [[RESULT2:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// X64-DAG: [[RESULT3:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// X64-DAG: store i32 [[RESULT0]], i32* [[ADDRPTR0]], align 4
// X64-DAG: store i32 [[RESULT1]], i32* [[ADDRPTR1]], align 4
// X64-DAG: store i32 [[RESULT2]], i32* [[ADDRPTR2]], align 4
// X64-DAG: store i32 [[RESULT3]], i32* [[ADDRPTR3]], align 4

#pragma intrinsic(__cpuidex)

void test__cpuidex(int cpuInfo[4], int function_id, int subfunction_id) {
  __cpuidex(cpuInfo, function_id, subfunction_id);
}
// X86-LABEL: define {{.*}} @test__cpuidex(i32* noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}})
// X86-DAG: [[ASMRESULTS:%[0-9]+]] = call { i32, i32, i32, i32 } asm "cpuid", "={ax},={bx},={cx},={dx},{ax},{cx}"
// X86-DAG: [[ADDRPTR0:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 0
// X86-DAG: [[ADDRPTR1:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 1
// X86-DAG: [[ADDRPTR2:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 2
// X86-DAG: [[ADDRPTR3:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 3
// X86-DAG: [[RESULT0:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// X86-DAG: [[RESULT1:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// X86-DAG: [[RESULT2:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// X86-DAG: [[RESULT3:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// X86-DAG: store i32 [[RESULT0]], i32* [[ADDRPTR0]], align 4
// X86-DAG: store i32 [[RESULT1]], i32* [[ADDRPTR1]], align 4
// X86-DAG: store i32 [[RESULT2]], i32* [[ADDRPTR2]], align 4
// X86-DAG: store i32 [[RESULT3]], i32* [[ADDRPTR3]], align 4

// X64-LABEL: define {{.*}} @test__cpuidex(i32* noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}})
// X64-DAG: [[ASMRESULTS:%[0-9]+]] = call { i32, i32, i32, i32 } asm "xchgq %rbx, ${1:q}\0Acpuid\0Axchgq %rbx, ${1:q}", "={ax},=r,={cx},={dx},0,2"
// X64-DAG: [[ADDRPTR0:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 0
// X64-DAG: [[ADDRPTR1:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 1
// X64-DAG: [[ADDRPTR2:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 2
// X64-DAG: [[ADDRPTR3:%[0-9]+]] = getelementptr inbounds i32, i32* %{{.*}}, i32 3
// X64-DAG: [[RESULT0:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 0
// X64-DAG: [[RESULT1:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 1
// X64-DAG: [[RESULT2:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 2
// X64-DAG: [[RESULT3:%[0-9]+]] = extractvalue { i32, i32, i32, i32 } [[ASMRESULTS]], 3
// X64-DAG: store i32 [[RESULT0]], i32* [[ADDRPTR0]], align 4
// X64-DAG: store i32 [[RESULT1]], i32* [[ADDRPTR1]], align 4
// X64-DAG: store i32 [[RESULT2]], i32* [[ADDRPTR2]], align 4
// X64-DAG: store i32 [[RESULT3]], i32* [[ADDRPTR3]], align 4
