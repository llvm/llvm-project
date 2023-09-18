// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

typedef void * id;

@protocol foo
@end

@interface CL
{
  id <foo> changeSource;
  CL <foo>* changeSource1;
}
@end

typedef struct x
{
   id <foo> changeSource;
} x;

