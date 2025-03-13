// RUN: %clang_cc1 -triple x86_64-linux-gnu -ast-dump %s | FileCheck %s

bool aaa;
constexpr const unsigned char ccc[] = { 5 };
constexpr const unsigned char ddd[1] = { 0 };
auto bbb = (aaa ? ddd : ccc);

// CHECK: DeclRefExpr {{.*}} 'const unsigned char[1]' {{.*}} 'ddd' 'const unsigned char[1]'
