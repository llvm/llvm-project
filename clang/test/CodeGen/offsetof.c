// RUN: %clang_cc1 %s -emit-llvm -o %t

// PR2910
struct sockaddr_un {
 unsigned char sun_len;
 char sun_path[104];
};

int test(int len) {
  return __builtin_offsetof(struct sockaddr_un, sun_path[len+1]);
}

