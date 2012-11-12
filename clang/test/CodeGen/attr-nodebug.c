// RUN: %clang_cc1 -g -emit-llvm -o %t %s
// RUN: not grep 'call void @llvm.dbg.func.start' %t

void t1() __attribute__((nodebug));

void t1()
{
  int a = 10;
  
  a++;
}

