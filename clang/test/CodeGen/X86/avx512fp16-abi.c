// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-feature +avx512fp16 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-feature +avx512fp16 -x c++ -std=c++11 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-CPP

struct half1 {
  _Float16 a;
};

struct half1 h1(_Float16 a) {
  // CHECK: define{{.*}}half @
  struct half1 x;
  x.a = a;
  return x;
}

struct half2 {
  _Float16 a;
  _Float16 b;
};

struct half2 h2(_Float16 a, _Float16 b) {
  // CHECK: define{{.*}}<2 x half> @
  struct half2 x;
  x.a = a;
  x.b = b;
  return x;
}

struct half3 {
  _Float16 a;
  _Float16 b;
  _Float16 c;
};

struct half3 h3(_Float16 a, _Float16 b, _Float16 c) {
  // CHECK: define{{.*}}<4 x half> @
  struct half3 x;
  x.a = a;
  x.b = b;
  x.c = c;
  return x;
}

struct half4 {
  _Float16 a;
  _Float16 b;
  _Float16 c;
  _Float16 d;
};

struct half4 h4(_Float16 a, _Float16 b, _Float16 c, _Float16 d) {
  // CHECK: define{{.*}}<4 x half> @
  struct half4 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  return x;
}

struct floathalf {
  float a;
  _Float16 b;
};

struct floathalf fh(float a, _Float16 b) {
  // CHECK: define{{.*}}<4 x half> @
  struct floathalf x;
  x.a = a;
  x.b = b;
  return x;
}

struct floathalf2 {
  float a;
  _Float16 b;
  _Float16 c;
};

struct floathalf2 fh2(float a, _Float16 b, _Float16 c) {
  // CHECK: define{{.*}}<4 x half> @
  struct floathalf2 x;
  x.a = a;
  x.b = b;
  x.c = c;
  return x;
}

struct halffloat {
  _Float16 a;
  float b;
};

struct halffloat hf(_Float16 a, float b) {
  // CHECK: define{{.*}}<4 x half> @
  struct halffloat x;
  x.a = a;
  x.b = b;
  return x;
}

struct half2float {
  _Float16 a;
  _Float16 b;
  float c;
};

struct half2float h2f(_Float16 a, _Float16 b, float c) {
  // CHECK: define{{.*}}<4 x half> @
  struct half2float x;
  x.a = a;
  x.b = b;
  x.c = c;
  return x;
}

struct floathalf3 {
  float a;
  _Float16 b;
  _Float16 c;
  _Float16 d;
};

struct floathalf3 fh3(float a, _Float16 b, _Float16 c, _Float16 d) {
  // CHECK: define{{.*}}{ <4 x half>, half } @
  struct floathalf3 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  return x;
}

struct half5 {
  _Float16 a;
  _Float16 b;
  _Float16 c;
  _Float16 d;
  _Float16 e;
};

struct half5 h5(_Float16 a, _Float16 b, _Float16 c, _Float16 d, _Float16 e) {
  // CHECK: define{{.*}}{ <4 x half>, half } @
  struct half5 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  x.e = e;
  return x;
}

struct float2 {
  struct {} s;
  float a;
  float b;
};

float pr51813(struct float2 s) {
  // CHECK-C: define{{.*}} @pr51813(<2 x float>
  // CHECK-CPP: define{{.*}} @_Z7pr518136float2(double {{.*}}, float
  return s.a;
}

struct float3 {
  float a;
  struct {} s;
  float b;
};

float pr51813_2(struct float3 s) {
  // CHECK-C: define{{.*}} @pr51813_2(<2 x float>
  // CHECK-CPP: define{{.*}} @_Z9pr51813_26float3(double {{.*}}, float
  return s.a;
}

struct shalf2 {
  struct {} s;
  _Float16 a;
  _Float16 b;
};

_Float16 sf2(struct shalf2 s) {
  // CHECK-C: define{{.*}} @sf2(<2 x half>
  // CHECK-CPP: define{{.*}} @_Z3sf26shalf2(double {{.*}}
  return s.a;
};

struct halfs2 {
  _Float16 a;
  struct {} s1;
  _Float16 b;
  struct {} s2;
};

_Float16 fs2(struct shalf2 s) {
  // CHECK-C: define{{.*}} @fs2(<2 x half>
  // CHECK-CPP: define{{.*}} @_Z3fs26shalf2(double {{.*}}
  return s.a;
};

struct fsd {
  float a;
  struct {};
  double b;
};

struct fsd pr52011(void) {
  // CHECK: define{{.*}} { float, double } @
  struct fsd x;
  return x;
}

struct hsd {
  _Float16 a;
  struct {};
  double b;
};

struct hsd pr52011_2(void) {
  // CHECK: define{{.*}} { half, double } @
  struct hsd x;
  return x;
}

struct hsf {
  _Float16 a;
  struct {};
  float b;
};

struct hsf pr52011_3(void) {
  // CHECK: define{{.*}} <4 x half> @
  struct hsf x;
  return x;
}

struct fds {
  float a;
  double b;
  struct {};
};

struct fds pr52011_4(void) {
  // CHECK-C: define{{.*}} { float, double } @pr52011_4
  // CHECK-CPP: define{{.*}} void @_Z9pr52011_4v({{.*}} sret
  struct fds x;
  return x;
}
