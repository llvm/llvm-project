// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

static const int GlobalConst = 42;
// CHECK: VarDecl{{.*}} used GlobalConst 'const int'
#pragma acc declare copyin(GlobalConst)

static const int CacheConst = 1;
// CHECK: VarDecl{{.*}} referenced CacheConst 'const int'

static const int CacheConstArr[10] = {};
// CHECK: VarDecl{{.*}} used CacheConstArr 'const int[10]'

void use(int (&Arr)[10]) {
// CHECK: FunctionDecl{{.*}} use 'void (int (&)[10])'
  static const int LocalConst = 2;
// CHECK: VarDecl{{.*}} used LocalConst 'const int'
#pragma acc parallel copyin(LocalConst)
  ;

#pragma acc loop
  for (int i = 0; i < 5; ++i) {
#pragma acc cache(Arr[CacheConst], CacheConstArr[1:2])
    ;
  }
}
