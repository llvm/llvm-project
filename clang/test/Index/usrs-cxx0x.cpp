template<typename ...Types>
struct tuple { };

void f(tuple<int, float, double>);

// RUN: c-index-test -test-load-source-usrs all -std=c++11 %s | FileCheck %s
// CHECK: usrs-cxx0x.cpp c:@ST>1#pT@tuple Extent=[1:1 - 2:17]
// CHECK: usrs-cxx0x.cpp c:@F@f#$@S@tuple>#p3Ifd# Extent=[4:1 - 4:34]
