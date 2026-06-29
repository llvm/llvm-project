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

template<typename T>
T pi = T(3.14);

template<>
int pi<int> = 3;

// Variable template declaration: should produce VarTemplate cursor.
// CHECK: VarTemplate=pi:17:3 (Definition)

// Variable template specialization: should report 1 argument.
// CHECK: VarDecl=pi:20:5 (Definition) [Specialization of pi:17:3] [Template arg 0: kind: 1, type: int]

template<typename T>
T tau = T(6.28);

template<>
float tau<float> = 6.28f;

template<typename U>
U tau<U*> = U(0);

// Variable template explicit specialization.
// CHECK: VarDecl=tau:32:7 (Definition) [Specialization of tau:29:3] [Template arg 0: kind: 1, type: float]

// Variable template partial specialization.
// CHECK: VarTemplatePartialSpecialization=tau:35:3 (Definition) [Specialization of tau:29:3] [Template arg 0: kind: 1, type: type-parameter-0-0 *]

template<int N>
int ival = N;

template<>
int ival<42> = 42;

// Variable template with NTTP: should report integral argument.
// CHECK: VarDecl=ival:47:5 (Definition) [Specialization of ival:44:5] [Template arg 0: kind: 4, intval: 42]
