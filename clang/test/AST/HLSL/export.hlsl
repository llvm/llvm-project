// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK:ExportDecl 0x{{[0-9a-f]+}} <{{.*}}> col:1
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> col:13 used f1 'void ()'
// CHECK:CompoundStmt 0x{{[0-9a-f]+}} <{{.*}}>
export void f1() {}

// CHECK:NamespaceDecl 0x{{[0-9a-f]+}} <{{.*}}>
// CHECK:ExportDecl 0x{{[0-9a-f]+}} <{{.*}}> col:3
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> col:15 used f2 'void ()'
// CHECK:CompoundStmt 0x{{[0-9a-f]+}} <{{.*}}>
namespace MyNamespace {
  export void f2() {}
}

// CHECK:ExportDecl 0x{{[0-9a-f]+}} <{{.*}}>
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> col:10 used f3 'void ()'
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> col:10 used f4 'void ()'
// CHECK:CompoundStmt 0x{{[0-9a-f]+}} <{{.*}}>
export {
    void f3() {}
    void f4() {}
}
