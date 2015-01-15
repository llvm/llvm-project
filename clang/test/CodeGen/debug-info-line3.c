// RUN: %clang_cc1 -g -S -emit-llvm %s -o - | FileCheck %s

void func(char c, char* d)
{
  *d = c + 1;
  return;
  

  
  
  
  
}

// CHECK: ret void, !dbg [[LINE:.*]]
// CHECK: [[LINE]] = !MDLocation(line: 6,
