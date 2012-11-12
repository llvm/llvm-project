// RUN: %clang_cc1 -emit-llvm %s -o -

typedef int (*_MD_Open64)(int oflag, ...);
_MD_Open64 _open64;
void PR_OpenFile(int mode) {
_open64(0, mode);
}
