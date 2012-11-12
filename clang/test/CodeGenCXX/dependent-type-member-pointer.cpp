// RUN: %clang_cc1 -emit-llvm-only -verify %s
// expected-no-diagnostics
// PR7736

template <class scriptmemberptr> int InitMember(scriptmemberptr);

template <class> 
struct contentmap
{
  static void InitDataMap()
  { InitMember(&contentmap::SizeHolder); }
  int SizeHolder;
};

void ReadFrom( )
{
  contentmap<int>::InitDataMap();
}

