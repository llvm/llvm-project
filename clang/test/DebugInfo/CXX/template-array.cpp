// RUN: %clang -emit-llvm -g -S %s -o -
// PR13531
template <typename>
struct unique_ptr {
  unique_ptr() {}
};

template <unsigned>
struct Vertex {};

void crash() // Asserts
{
  unique_ptr<Vertex<2>[]> v = unique_ptr<Vertex<2>[]>();
}
