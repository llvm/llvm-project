// RUN: %clang_cc1 -std=c++20 -fmodules-ts %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
module;
# 4 __FILE__ 1
namespace Outer::Inner {
class X;
// CHECK-DAG: void @_ZN5Outer5Inner3BarERNS0_1XE(
void Bar (X &) {}
} // namespace Outer::Inner
# 10 "" 2
export module FOO;	
namespace Outer {
class Y;
namespace Inner {
// CHECK-DAG: void @_ZN5Outer5InnerW3FOO2FnERNS0_1XERNS_S1_1YE(
void Fn (X &, Y &){}  // #1
} // namespace Inner
} // namespace Outer
