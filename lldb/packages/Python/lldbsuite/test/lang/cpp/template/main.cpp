//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <tuple>

template <int Arg>
class TestObj
{
public:
  int getArg()
  {
    return Arg;
  }
};

//----------------------------------------------------------------------
// Define a template class that we can specialize with an enumeration
//----------------------------------------------------------------------
enum class EnumType
{
    Member,
    Subclass
};

template <EnumType Arg> class EnumTemplate;
                                          
//----------------------------------------------------------------------
// Specialization for use when "Arg" is "EnumType::Member"
//----------------------------------------------------------------------
template <>
class EnumTemplate<EnumType::Member> 
{
public:
    EnumTemplate(int m) :
        m_member(m)
    {
    }

    int getMember() const
    {
        return m_member;
    }

protected:
    int m_member;
};

//----------------------------------------------------------------------
// Specialization for use when "Arg" is "EnumType::Subclass"
//----------------------------------------------------------------------
template <>
class EnumTemplate<EnumType::Subclass> : 
    public EnumTemplate<EnumType::Member> 
{
public:
    EnumTemplate(int m) : EnumTemplate<EnumType::Member>(m)
    {
    }    
};

template <typename FLOAT> struct T1 { FLOAT f = 1.5; };
template <typename FLOAT> struct T2 { FLOAT f = 2.5; int i = 42; };
template <typename FLOAT, template <typename> class ...Args> class C { std::tuple<Args<FLOAT>...> V; };

int main(int argc, char **argv)
{
  TestObj<1> testpos;
  TestObj<-1> testneg;
  EnumTemplate<EnumType::Member> member(123);
  EnumTemplate<EnumType::Subclass> subclass(123*2);
  C<float, T1> c1;
  C<double, T1, T2> c2;
  return testpos.getArg() - testneg.getArg() + member.getMember()*2 - subclass.getMember(); // Breakpoint 1
}
