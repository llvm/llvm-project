// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | not grep define
// PR7415
class X {
  template <class Dummy> struct COMTypeInfo {
    static const int kIID;
  };
  static const int& GetIID() {return COMTypeInfo<int>::kIID;}
};
template <class Dummy> const int X::COMTypeInfo<Dummy>::kIID = 10;



