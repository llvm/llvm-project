// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple x86_64-pc-linux -include-pch %t \
// RUN:            -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

void TestUnattributedMVF(void);
// CHECK:      FunctionDecl 0x{{[^ ]*}} <{{.*}}> col:{{[0-9]*}} multiversion TestUnattributedMVF
__attribute__((target("default"))) void TestUnattributedMVF(void);
// CHECK:      FunctionDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <{{.*}}> col:{{[0-9]*}} multiversion TestUnattributedMVF

__attribute__((target("mmx"))) void TestNonMVF(void);
// CHECK:      FunctionDecl 0x{{[^ ]*}} <{{.*}}> col:{{[0-9]*}} TestNonMVF

__attribute__((target("mmx"))) void TestRetroMVF(void);
// CHECK:      FunctionDecl 0x{{[^ ]*}} <{{.*}}> col:{{[0-9]*}} multiversion TestRetroMVF
__attribute__((target("sse"))) void TestRetroMVF(void);
// CHECK:      FunctionDecl 0x{{[^ ]*}} <{{.*}}> col:{{[0-9]*}} multiversion TestRetroMVF
