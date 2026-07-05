// RUN: c-index-test core -print-source-symbols -include-locals -- %s | FileCheck %s

template <template <typename> class A> class B { typedef A<int> A_int; };
// CHECK: [[@LINE-1]]:46 | class(Gen)/C++ | B | c:@ST>1#t>1#T@B | <no-cgname> | Def | rel: 0
