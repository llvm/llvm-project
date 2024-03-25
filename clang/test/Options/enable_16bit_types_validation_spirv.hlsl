// RUN: not %clang_cc1 -internal-isystem D:\llvm-project\build\x64-Release\lib\clang\19\include -nostdsysteminc -triple spirv-vulkan-library -x hlsl -std=hlsl2016 -fnative-half-type -emit-llvm -disable-llvm-passes  -o - %s 2>&1 | FileCheck %s --check-prefix=SPIRV

// SPIRV: error: fnative-half-type option only valid when hlsl language standard version is >= 2021

[numthreads(1,1,1)]
void main()
{
  return;
}

