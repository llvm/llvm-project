// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol MyProtocol
@end

id<MyProtocol> obj_p = 0;

int main(void)
{
  obj_p = 0;
}

