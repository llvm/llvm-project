// RUN: %clang_cc1 -fsyntax-only -verify %s

extern void foo(void);

@protocol MyProtocol @end

@interface MyClass @end

int main(void)
{
  MyClass <MyProtocol> *obj_p;
  MyClass *obj_cp;

  obj_cp = obj_p;  
  obj_p = obj_cp;	// expected-warning {{incompatible pointer types assigning to 'MyClass<MyProtocol> *' from 'MyClass *'}}

  if (obj_cp == obj_p)
    foo();

  if (obj_p == obj_cp)
    foo();

}


