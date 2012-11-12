// Test this without pch.
// RUN: %clang_cc1 -fsyntax-only -emit-llvm-only %s

// Test with pch.
// RUN: touch %t.empty.cpp
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -emit-llvm-only %t.empty.cpp 

// rdar://10830559

#pragma ms_struct on

template< typename T >
class Templated
{
public:
   struct s;
};


class Foo
{
private:

   class Bar
   {
   private:
      class BarTypes { public: virtual void Func(); }; 
      class BarImpl {};
      friend class Foo;
   };
   
   
   friend class Templated< Bar::BarImpl >::s;
};
