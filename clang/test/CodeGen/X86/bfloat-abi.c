// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-feature +sse2 < %s | FileCheck %s --check-prefixes=CHECK

struct bfloat1 {
  __bf16 a;
};

struct bfloat1 h1(__bf16 a) {
  // CHECK: define{{.*}}bfloat @
  struct bfloat1 x;
  x.a = a;
  return x;
}

struct bfloat2 {
  __bf16 a;
  __bf16 b;
};

struct bfloat2 h2(__bf16 a, __bf16 b) {
  // CHECK: define{{.*}}<2 x bfloat> @
  struct bfloat2 x;
  x.a = a;
  x.b = b;
  return x;
}

struct bfloat3 {
  __bf16 a;
  __bf16 b;
  __bf16 c;
};

struct bfloat3 h3(__bf16 a, __bf16 b, __bf16 c) {
  // CHECK: define{{.*}}<4 x bfloat> @
  struct bfloat3 x;
  x.a = a;
  x.b = b;
  x.c = c;
  return x;
}

struct bfloat4 {
  __bf16 a;
  __bf16 b;
  __bf16 c;
  __bf16 d;
};

struct bfloat4 h4(__bf16 a, __bf16 b, __bf16 c, __bf16 d) {
  // CHECK: define{{.*}}<4 x bfloat> @
  struct bfloat4 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  return x;
}

struct floatbfloat {
  float a;
  __bf16 b;
};

struct floatbfloat fh(float a, __bf16 b) {
  // CHECK: define{{.*}}<4 x half> @
  struct floatbfloat x;
  x.a = a;
  x.b = b;
  return x;
}

struct floatbfloat2 {
  float a;
  __bf16 b;
  __bf16 c;
};

struct floatbfloat2 fh2(float a, __bf16 b, __bf16 c) {
  // CHECK: define{{.*}}<4 x half> @
  struct floatbfloat2 x;
  x.a = a;
  x.b = b;
  x.c = c;
  return x;
}

struct bfloatfloat {
  __bf16 a;
  float b;
};

struct bfloatfloat hf(__bf16 a, float b) {
  // CHECK: define{{.*}}<4 x half> @
  struct bfloatfloat x;
  x.a = a;
  x.b = b;
  return x;
}

struct bfloat2float {
  __bf16 a;
  __bf16 b;
  float c;
};

struct bfloat2float h2f(__bf16 a, __bf16 b, float c) {
  // CHECK: define{{.*}}<4 x bfloat> @
  struct bfloat2float x;
  x.a = a;
  x.b = b;
  x.c = c;
  return x;
}

struct floatbfloat3 {
  float a;
  __bf16 b;
  __bf16 c;
  __bf16 d;
};

struct floatbfloat3 fh3(float a, __bf16 b, __bf16 c, __bf16 d) {
  // CHECK: define{{.*}}{ <4 x half>, bfloat } @
  struct floatbfloat3 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  return x;
}

struct bfloat5 {
  __bf16 a;
  __bf16 b;
  __bf16 c;
  __bf16 d;
  __bf16 e;
};

struct bfloat5 h5(__bf16 a, __bf16 b, __bf16 c, __bf16 d, __bf16 e) {
  // CHECK: define{{.*}}{ <4 x bfloat>, bfloat } @
  struct bfloat5 x;
  x.a = a;
  x.b = b;
  x.c = c;
  x.d = d;
  x.e = e;
  return x;
}
