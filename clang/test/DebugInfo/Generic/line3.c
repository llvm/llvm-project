// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s

void func(char c, char* d)
{
  *d = c + 1;
  return;
  

  
  
  
  
}

// CHECK: ret void, !dbg [[LINE:.*]]
// CHECK: [[LINE]] = !DILocation(line: 6,
