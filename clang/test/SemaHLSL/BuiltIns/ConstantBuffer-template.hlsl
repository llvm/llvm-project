// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -fsyntax-only -verify %s

struct T { // expected-note 3 {{candidate constructor}}
    int a;
};

ConstantBuffer<T> c;

RWBuffer<int> b;

template<class Tm>
void foo(Tm t) {
    b[0] = t.a;
}

[numthreads(1,1,1)]
void main() {
  // An implicit conversion from ConstantBuffer<T> to T should work.
  // It is not implemented yet
  // (https://github.com/llvm/llvm-project/issues/153055).
  // expected-error@+1 {{no viable constructor copying variable of type 'const hlsl_constant T'}}
  T t = c;
}
