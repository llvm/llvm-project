// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s -check-prefix=ALL -check-prefix=NORMAL
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - -target-abi elfv1-qpx %s | FileCheck %s -check-prefix=ALL -check-prefix=QPX

typedef float v4sf __attribute__((vector_size(16)));
typedef double v4df __attribute__((vector_size(32)));

struct ssf { v4sf v; };
struct sdf { v4df v; };

struct ssf2 { v4sf v[2]; };
struct sdf2 { v4df v[2]; };

v4sf foo1(struct ssf a, v4sf b, struct ssf2 c) {
  return a.v + b;
}

// ALL-LABEL: define <4 x float> @foo1(<4 x float> inreg %a.coerce, <4 x float> %b, [2 x i128] %c.coerce)
// ALL: ret <4 x float>

v4df foo2(struct sdf a, v4df b, struct sdf2 c) {
  return a.v + b;
}

// QPX-LABEL: define <4 x double> @foo2(<4 x double> inreg %a.coerce, <4 x double> %b, [2 x i256] %c.coerce)
// QPX: ret <4 x double>

// NORMAL-LABEL: define void @foo2(<4 x double>* noalias sret(<4 x double>) align 32 %agg.result, [2 x i128] %a.coerce, <4 x double>* %0, [4 x i128] %c.coerce)
// NORMAL: ret void

