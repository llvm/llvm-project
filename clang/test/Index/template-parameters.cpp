// Test template argument pack expansion.
// RUN: c-index-test -test-load-source all -fno-delayed-template-parsing %s | FileCheck %s

template<typename T, typename... Ts>
struct Variadic {};

template class Variadic<int, float, double>;
template class Variadic<int>;

// Pack with 3 args: should report 3 (not 2).
// CHECK: StructDecl=Variadic:7:16 (Definition) [Specialization of Variadic:5:8] [Template arg 0: kind: 1, type: int] [Template arg 1: kind: 1, type: float] [Template arg 2: kind: 1, type: double]

// Empty pack: should report 1 (just T, pack contributes 0).
// CHECK: StructDecl=Variadic:8:16 (Definition) [Specialization of Variadic:5:8] [Template arg 0: kind: 1, type: int]
