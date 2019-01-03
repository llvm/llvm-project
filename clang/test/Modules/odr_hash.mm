// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x objective-c++ %t/Inputs/first.h -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only -x objective-c++ %t/Inputs/second.h -fblocks -fobjc-arc

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c++ -I%t/Inputs -verify %s -fblocks -fobjc-arc

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST) || defined(SECOND)
@protocol P1
@end

@protocol P2
@end

@interface I1
@end

@interface I2 : I1
@end

@interface Interface1 <T : I1 *> {
@public
  T x; // FIXME: align with upstream (rdar://43906928).
}
@end

@interface Interface2 <T : I1 *>
@end

@interface Interface3 <T : I1 *>
@end

#endif

#if defined(FIRST)
struct S {
  Interface1 *I;
  decltype(I->x) x;
  int y;
};
#elif defined(SECOND)
struct S {
  Interface1 *I;
  decltype(I->x) x;
  bool y;
};
#else
S s;
// expected-error@second.h:* {{'S::y' from module 'SecondModule' is not present in definition of 'S' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'y' does not match}}
#endif

namespace Types {
namespace Attributed {
#if defined(FIRST)
void invalid1() {
  static double __attribute((objc_gc(strong))) *x;
}
void invalid2() {
  static int __attribute((objc_gc(strong))) *x;
}
void valid() {
  static int __attribute((objc_gc(strong))) *x;
}
#elif defined(SECOND)
void invalid1() {
  static int __attribute((objc_gc(strong))) *x;
}
void invalid2() {
  static int __attribute((objc_gc(weak))) *x;
}
void valid() {
  static int __attribute((objc_gc(strong))) *x;
}
#else
auto function1 = invalid1;
// expected-error@second.h:* {{Types::Attributed::invalid1' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
auto function2 = invalid2;
// expected-error@second.h:* {{'Types::Attributed::invalid2' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
auto function3 = valid;
#endif
}  // namespace Attributed

namespace BlockPointer {
#if defined(FIRST)
void invalid1() {
  void (^x)(int);
}
void invalid2() {
  void (^x)(int);
}
void invalid3() {
  void (^x)(int);
}
void invalid4() {
  void (^x)(int);
}
void valid() {
  void (^x1)(int);
  int (^x2)(int);
  void (^x3)(int, int);
  void (^x4)(short);
}
#elif defined(SECOND)
void invalid1() {
  void (^x)();
}
void invalid2() {
  void (^x)(int, int);
}
void invalid3() {
  int (^x)(int);
}
void invalid4() {
  void (^x)(float);
}
void valid() {
  void (^x1)(int);
  int (^x2)(int);
  void (^x3)(int, int);
  void (^x4)(short);
}
#else
auto function1 = invalid1;
// expected-error@second.h:* {{'Types::BlockPointer::invalid1' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
auto function2 = invalid2;
// expected-error@second.h:* {{'Types::BlockPointer::invalid2' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
auto function3 = invalid3;
// expected-error@second.h:* {{'Types::BlockPointer::invalid3' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
auto function4 = invalid4;
// expected-error@second.h:* {{'Types::BlockPointer::invalid4' has different definitions in different modules; definition in module 'SecondModule' first difference is function body}}
// expected-note@first.h:* {{but in 'FirstModule' found a different body}}
auto function5 = valid;
#endif
}  // namespace BlockPointer

namespace ObjCObject {
#if defined(FIRST)
struct Invalid1 {
  using T = Interface2<I1*>;
};
struct Invalid2 {
  using T = Interface2<I1*>;
};
struct Invalid3 {
  using T = Interface2<P1, P1>;
};
struct Invalid4 {
  using T = Interface2<P1>;
};
struct Valid {
  using T1 = Interface2<I1*>;
  using T2 = Interface3<I1*>;
  using T3 = Interface2<P1>;
  using T4 = Interface3<P1, P2>;
  using T5 = __kindof Interface2;
};
#elif defined(SECOND)
struct Invalid1 {
  using T = Interface3<I1*>;
};
struct Invalid2 {
  using T = Interface2<I2*>;
};
struct Invalid3 {
  using T = Interface2<P1>;
};
struct Invalid4 {
  using T = Interface2<P2>;
};
struct Valid {
  using T1 = Interface2<I1*>;
  using T2 = Interface3<I1*>;
  using T3 = Interface2<P1>;
  using T4 = Interface3<P1, P2>;
  using T5 = __kindof Interface2;
};
#else
Invalid1 i1;
// expected-error@first.h:* {{'Types::ObjCObject::Invalid1::T' from module 'FirstModule' is not present in definition of 'Types::ObjCObject::Invalid1' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'T' does not match}}
Invalid2 i2;
// expected-error@first.h:* {{'Types::ObjCObject::Invalid2::T' from module 'FirstModule' is not present in definition of 'Types::ObjCObject::Invalid2' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'T' does not match}}
Invalid3 i3;
// expected-error@second.h:* {{'Types::ObjCObject::Invalid3' has different definitions in different modules; first difference is definition in module 'SecondModule' found type alias 'T' with underlying type 'Interface2<P1>'}}
// expected-note@first.h:* {{but in 'FirstModule' found type alias 'T' with different underlying type 'Interface2<P1,P1>'}}
Invalid4 i4;
// expected-error@first.h:* {{'Types::ObjCObject::Invalid4::T' from module 'FirstModule' is not present in definition of 'Types::ObjCObject::Invalid4' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'T' does not match}}
Valid v;
#endif
}  // namespace VisitObjCObject
}  // namespace Types

#if defined(FIRST)
@interface Interface4 <T : I1 *> {
@public
  T x; // FIXME: align with upstream (rdar://43906928).
}
@end
@interface Interface5 <T : I1 *> {
@public
  T x; // FIXME: align with upstream (rdar://43906928).
}
@end
@interface Interface6 <T1 : I1 *, T2 : I2 *> {
@public
  T1 x;
}
@end
#elif defined(SECOND)
@interface Interface4 <T : I1 *> {
@public
  T x; // FIXME: align with upstream (rdar://43906928).
}
@end
@interface Interface5 <T : I1 *> {
@public
  T x; // FIXME: align with upstream (rdar://43906928).
}
@end
@interface Interface6 <T1 : I1 *, T2 : I2 *> {
@public
  T2 x;
}
@end
#endif

namespace Types {
namespace ObjCTypeParam {
#if defined(FIRST) || defined(SECOND)
struct Invalid1 {
  Interface4 *I;
  decltype(I->x) x;
};
struct Invalid2 {
  Interface5 *I;
  decltype(I->x) x;
};
struct Invalid3 {
  Interface6 *I;
  decltype(I->x) x;
};
#else
Invalid1 i1;
Invalid2 i2;
Invalid3 i3;
// expected-error@first.h:* {{'Types::ObjCTypeParam::Invalid3::x' from module 'FirstModule' is not present in definition of 'Types::ObjCTypeParam::Invalid3' in module 'SecondModule'}}
// expected-note@second.h:* {{declaration of 'x' does not match}}
#endif

}  // namespace ObjCTypeParam
}  // namespace Types

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
