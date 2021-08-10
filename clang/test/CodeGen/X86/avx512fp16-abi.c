// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-feature +avx512fp16 < %s | FileCheck %s --check-prefixes=CHECK

struct half1 {
  _Float16 a;
};

struct half1 h1(_Float16 a) {
  // CHECK: define{{.*}}half @h1
  struct half1 x;
  x.a = a;
  return x;
}

struct half2 {
  _Float16 a;
  _Float16 b;
};

struct half2 h2(_Float16 a, _Float16 b) {
  // CHECK: define{{.*}}<2 x half> @h2
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
  // CHECK: define{{.*}}<4 x half> @h3
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
  // CHECK: define{{.*}}<4 x half> @h4
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
  // CHECK: define{{.*}}<4 x half> @fh
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
  // CHECK: define{{.*}}<4 x half> @fh2
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
  // CHECK: define{{.*}}<4 x half> @hf
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
  // CHECK: define{{.*}}<4 x half> @h2f
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
  // CHECK: define{{.*}}{ <4 x half>, half } @fh3
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
  // CHECK: define{{.*}}{ <4 x half>, half } @h5
  struct half5 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  x.e = e;
  return x;
}
