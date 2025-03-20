// RUN: not %clang_cc1 -internal-isystem D:\llvm-project\build\x64-Release\lib\clang\19\include -nostdsysteminc -triple spirv-vulkan-library -x hlsl -std=hlsl2016 -fnative-half-type -emit-llvm -disable-llvm-passes  -o - %s 2>&1 | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -internal-isystem D:\llvm-project\build\x64-Release\lib\clang\19\include -nostdsysteminc -triple spirv-vulkan-library -x hlsl -std=hlsl2021 -fnative-half-type -emit-llvm -disable-llvm-passes  -o - %s 2>&1 | FileCheck %s --check-prefix=valid

// SPIRV: error: '-fnative-half-type' option requires target HLSL Version >= 2018, but HLSL Version is 'hlsl2016'

// valid: "spirv-unknown-vulkan-library"
// valid: define spir_func void @{{.*main.*}}() #0 {

[numthreads(1,1,1)]
void main()
{
  return;
}

