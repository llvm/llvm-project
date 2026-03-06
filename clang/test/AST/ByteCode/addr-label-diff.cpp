// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o -                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - -fexperimental-new-constant-interpreter | FileCheck %s


// Make sure we don't try to fold this either.
// CHECK: @_ZZ23UnfoldableAddrLabelDiffvE1x = internal global i128 0
void UnfoldableAddrLabelDiff() { static __int128_t x = (long)&&a-(long)&&b; a:b:return;}

// CHECK: @_ZZ24UnfoldableAddrLabelDiff2vE1x = internal global i16 0
void UnfoldableAddrLabelDiff2() { static short x = (long)&&a-(long)&&b; a:b:return;}


// But make sure we do fold this.
// CHECK: @_ZZ21FoldableAddrLabelDiffvE1x = internal global i64 sub (i64 ptrtoint (ptr blockaddress(@_Z21FoldableAddrLabelDiffv
void FoldableAddrLabelDiff() { static long x = (long)&&a-(long)&&b; a:b:return;}

