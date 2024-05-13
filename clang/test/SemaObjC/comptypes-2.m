// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#define nil (void *)0;
#define Nil (void *)0;

@protocol MyProtocol
- (void) foo;
@end

@interface MyClass
@end

int main(void)
{
  id obj = nil;
  id<MyProtocol> obj_p = nil;
  MyClass *obj_c = nil;
  Class obj_C = Nil;

  /* All these casts should generate no warnings.  */

  obj = (id)obj_p;
  obj = (id)obj_c;
  obj = (id)obj_C;
  obj_c = (MyClass *)obj;
  obj_c = (MyClass *)obj_p;
  obj_c = (MyClass *)obj_C;
  obj_p = (id<MyProtocol>)obj;
  obj_p = (id<MyProtocol>)obj_c;
  obj_p = (id<MyProtocol>)obj_C;
  obj_C = (Class)obj;
  obj_C = (Class)obj_p;
  obj_C = (Class)obj_c;
  

  return 0;
}
