//RUN: %clang_cc1 -std=c++20 -ast-dump %s | FileCheck %s

export module A;

int inTheInterface();
// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> {{.*}}inTheInterface 'int ()'

module :private;
// CHECK: PrivateModuleFragmentDecl 0x{{[0-9a-f]+}} <{{.*}}>

int inThePrivateFragment();
// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> {{.*}}inThePrivateFragment 'int ()'
