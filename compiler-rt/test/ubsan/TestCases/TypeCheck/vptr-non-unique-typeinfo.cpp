// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -I%p/Helpers -g %s -fPIC -shared -o %t-lib.so -DBUILD_SO
// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -I%p/Helpers -g %s -O3 -o %t %t-lib.so
// RUN: %run %t
//
// REQUIRES: cxxabi

struct X {
  virtual ~X() {}
};
X *libCall();

#ifdef BUILD_SO

X *libCall() {
  return new X;
}

#else

int main() {
  X *px = libCall();
  delete px;
}

#endif
