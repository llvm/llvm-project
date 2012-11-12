// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

// CHECK: "\01?LRef@@YAXAAH@Z"
void LRef(int& a) { }

// CHECK: "\01?RRef@@YAH$$QAH@Z"
int RRef(int&& a) { return a; }

// CHECK: "\01?Null@@YAX$$T@Z"
namespace std { typedef decltype(__nullptr) nullptr_t; }
void Null(std::nullptr_t) {}
