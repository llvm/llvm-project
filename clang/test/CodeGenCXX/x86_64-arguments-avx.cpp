// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-feature +avx | FileCheck %s

namespace test1 {
typedef double __m256d __attribute__((__vector_size__(32)));

class PR22753 {
public:
  __m256d data;
};

// CHECK: define{{.*}} <4 x double> @_ZN5test14testENS_7PR22753E(<4 x double>
PR22753 test(PR22753 x) {
  return x;
}
}

namespace test2 {
typedef double __m128d __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));
typedef double __m256d __attribute__((__vector_size__(32)));
typedef float __m256 __attribute__((__vector_size__(32)));

union U1 {
  __m128  v1;
  __m128d v2;
};

union UU1 {
  union U1;
  __m128d v3;
};

// CHECK: define{{.*}} <2 x double> @_ZN5test27PR23082ENS_3UU1E(<2 x double>
UU1 PR23082(UU1 x) {
  return x;
}

union U2 {
  __m256  v1;
  __m256d v2;
};

union UU2 {
  union U2;
  __m256d v3;
};

// CHECK: define{{.*}} <4 x double> @_ZN5test27PR23082ENS_3UU2E(<4 x double>
UU2 PR23082(UU2 x) {
  return x;
}
}

namespace test3 {
union U {
  __attribute__((__vector_size__(32))) float f1;
  int f2;
};
// CHECK: define{{.*}} i32 @_ZN5test31fENS_1UE({{.*}}* noundef byval({{.*}}) align 32
int f(U u) { return u.f2; }
}
