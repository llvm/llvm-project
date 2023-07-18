// RUN: %clang_cc1 -std=c++11 -isystem %S/Inputs %s -verify
// expected-no-diagnostics
// rdar://18295240

#include <override-system-header.h>

struct A
{
  virtual void x();
  END_COM_MAP;
  IFACEMETHOD(Initialize)();
};
 
struct B : A
{
  virtual void x() override;
  END_COM_MAP;
  IFACEMETHOD(Initialize)();
};
