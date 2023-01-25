// RUN: %clang_cc1 -Wno-implicit-function-declaration -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: grep -F 'declare ptr @objc_msgSend(ptr noundef, ptr noundef, ...)' %t

typedef struct objc_selector *SEL;
id f0(id x, SEL s) {
  return objc_msgSend(x, s);
}
