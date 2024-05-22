// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -x c++ \
// RUN:  -emit-llvm %s -o -| FileCheck %s

extern "C" int printf(const char*,...);
class PrintfArg
{
public:
  PrintfArg();
  PrintfArg(const char* s);

  // compiler crash fixed if this destructor removed
  ~PrintfArg() {int x; printf("ddd\n");  }
};

void devif_Warning(const char* fmt, PrintfArg arg1 = PrintfArg());
// CHECK-NOT: invoke void @llvm.seh.scope.begin()
// CHECK-NOT: invoke void @llvm.seh.scope.end()
unsigned myfunc(unsigned index)
{
  devif_Warning("");
  return 0;
}
