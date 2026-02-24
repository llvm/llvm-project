// RUN: c-index-test -test-load-source all %s -std=gnu++20 | FileCheck %s

template<typename T>
concept Decrementable = requires (T t){ --t; };

auto i = 42;
// CHECK: index-auto.cpp:[[@LINE-1]]:6: VarDecl=i:[[@LINE-1]]:6 (Definition) Extent=[[[@LINE-1]]:1 - [[@LINE-1]]:12]

auto foo(){ return 42;}
// CHECK: index-auto.cpp:[[@LINE-1]]:6: FunctionDecl=foo:[[@LINE-1]]:6 (Definition) Extent=[[[@LINE-1]]:1 - [[@LINE-1]]:24]

Decrementable auto j = 43;
// CHECK: index-auto.cpp:[[@LINE-1]]:20: VarDecl=j:[[@LINE-1]]:20 (Definition) Extent=[[[@LINE-1]]:1 - [[@LINE-1]]:26]
// CHECK: index-auto.cpp:[[@LINE-2]]:1: TemplateRef=Decrementable:4:9 Extent=[[[@LINE-2]]:1 - [[@LINE-2]]:14]

Decrementable auto bar() { return 43; }
// CHECK: index-auto.cpp:[[@LINE-1]]:20: FunctionDecl=bar:[[@LINE-1]]:20 (Definition) Extent=[[[@LINE-1]]:1 - [[@LINE-1]]:40]
// CHECK: index-auto.cpp:[[@LINE-2]]:1: TemplateRef=Decrementable:4:9 Extent=[[[@LINE-2]]:1 - [[@LINE-2]]:14]
