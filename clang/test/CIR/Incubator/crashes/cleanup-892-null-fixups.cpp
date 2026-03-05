// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Test for UNREACHABLE at CIRGenCleanup.cpp:892
// Null fixups popping not yet implemented
//
// This test triggers the error:
// "UNREACHABLE executed at CIRGenCleanup.cpp:892!"
//
// Original failure: cleanup_892 from LLVM build
// Reduced from /tmp/Regex-8cd677.cpp

inline namespace a {
class c {
public:
  template <typename b> c(b);
  ~c();
};
} // namespace a
class d {
  c e() const;
};
class aj {
public:
  ~aj();
} an;
c d::e() const {
  aj ao;
  return an;
  c(0);
}
