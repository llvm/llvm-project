// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// CIR module verification error before passes
// Location: Module verification
//
// Original failure: verification_error from LLVM build
// Reduced from /tmp/Errno-48253a.cpp

inline namespace a {
class b {
public:
  ~b();
};
} // namespace a
b c() {
  b d;
  if (0)
    return d;
}
