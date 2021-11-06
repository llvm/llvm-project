// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: grep -F 'declare i8* @objc_msgSend(i8*, i8*, ...)' %t

typedef struct objc_selector *SEL;
id f0(id x, SEL s) {
  return objc_msgSend(x, s);
}
