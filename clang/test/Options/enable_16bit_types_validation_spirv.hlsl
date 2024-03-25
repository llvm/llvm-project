// RUN: %clang_cc1 -internal-isystem D:\llvm-project\build\x64-Release\lib\clang\19\include -nostdsysteminc -triple spirv-vulkan-library -x hlsl -std=hlsl2016 -emit-llvm -disable-llvm-passes  -o - %s | FileCheck %s --check-prefix=SPIRV
// kRUN: %clang_dxc -T lib_6_4 -HV 2016 -enable-16bit-types %s | FileCheck %s --check-prefix=SPIRV
// SPIRV: error: enable_16bit_types option only valid when target shader model [-T] is >= 6.2 and HLSL Version [-HV] is >= 2021

[numthreads(1,1,1)]
void main()
{
  return;
}

