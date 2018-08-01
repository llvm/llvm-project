
int capturedFunc(void (*fp)(int x)) {
  auto fp2 = fp;
  capturedFunc(fp);
  return capturedFunc(fp2);
}

// CHECK1: extracted(void (*fp)(int))

typedef void (*FuncTypedef)();

int capturedFuncTypedef(FuncTypedef fp) {
  return capturedFuncTypedef(fp);
}
// CHECK1: extracted(FuncTypedef fp)

// RUN: clang-refactor-test perform -action extract -selected=%s:4:3-4:19 -selected=%s:13:10-13:33 %s | FileCheck --check-prefix=CHECK1 %s
