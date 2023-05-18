template<typename ...Types>
struct tuple { };

void f(tuple<int, float, double>);

class TestCls {
  void meth() &;
  void meth() &&;
  void meth(int&&);
};

// RUN: c-index-test -test-load-source-usrs all -std=c++11 %s | FileCheck %s
// CHECK: usrs-cxx0x.cpp c:@ST>1#pT@tuple Extent=[1:1 - 2:17]
// CHECK: usrs-cxx0x.cpp c:@F@f#$@S@tuple>#p3Ifd# Extent=[4:1 - 4:34]

// CHECK: usrs-cxx0x.cpp c:@S@TestCls@F@meth#& Extent=[7:3 - 7:16]
// CHECK: usrs-cxx0x.cpp c:@S@TestCls@F@meth#&& Extent=[8:3 - 8:17]
// CHECK: usrs-cxx0x.cpp c:@S@TestCls@F@meth#&&I# Extent=[9:3 - 9:19]
