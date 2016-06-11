//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

template <int Arg>
class TestObj
{
public:
  int getArg()
  {
    return Arg;
  }
};

int main(int argc, char **argv)
{
  TestObj<1> testpos;
  TestObj<-1> testneg;
  return testpos.getArg() - testneg.getArg(); // Breakpoint 1
}
