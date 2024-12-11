
#include <array-param-sys.h>

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include | FileCheck %S/include/array-param-sys.h --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --implicit-check-not RecoveryExpr

void foo(int * __counted_by(size) arr, int size) {
    funcInSDK(size, arr);
}

