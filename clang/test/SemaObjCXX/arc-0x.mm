// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -fobjc-weak -verify -fblocks -fobjc-exceptions %s

// "Move" semantics, trivial version.
void move_it(__strong id &&from) {
  id to = static_cast<__strong id&&>(from);
}

// Deduction with 'auto'.
@interface A
+ alloc;
- init;
@end

// <rdar://problem/12031870>: don't warn about this
extern "C" A* MakeA();

// Ensure that deduction works with lifetime qualifiers.
void deduction(id obj) {
  auto a = [[A alloc] init];
  __strong A** aPtr = &a;

  auto a2([[A alloc] init]);
  __strong A** aPtr2 = &a2;

  __strong id *idp = new auto(obj);

  __strong id array[17];
  for (auto x : array) { // expected-warning{{'auto' deduced as 'id' in declaration of 'x'}}
    __strong id *xPtr = &x;
  }

  @try {
  } @catch (auto e) { // expected-error {{'auto' not allowed in exception declaration}}
  }
}

// rdar://problem/11068137
void test1a() {
  __autoreleasing id p; // expected-note 2 {{'p' declared here}}
  (void) [&p] {};
  (void) [p] {}; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
  (void) [=] { (void) p; }; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
}
void test1b() {
  __autoreleasing id v;
  __autoreleasing id &p = v; // expected-note 2 {{'p' declared here}}
  (void) [&p] {};
  (void) [p] {}; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
  (void) [=] { (void) p; }; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
}
void test1c() {
  __autoreleasing id v; // expected-note {{'v' declared here}}
  __autoreleasing id &p = v;
  (void) ^{ (void) p; };
  (void) ^{ (void) v; }; // expected-error {{cannot capture __autoreleasing variable in a block}}
}


// <rdar://problem/11319689>
// warn when initializing an 'auto' variable with an 'id' initializer expression

void testAutoId(id obj) {
  auto x = obj; // expected-warning{{'auto' deduced as 'id' in declaration of 'x'}}
}

@interface Array
+ (instancetype)new;
- (id)objectAtIndex:(int)index;
@end

// ...but don't warn if it's coming from a template parameter.
template<typename T, int N>
void autoTemplateFunction(T param, id obj, Array *arr) {
  auto x = param; // no-warning
  auto y = obj; // expected-warning{{'auto' deduced as 'id' in declaration of 'y'}}
  auto z = [arr objectAtIndex:N]; // expected-warning{{'auto' deduced as 'id' in declaration of 'z'}}
}

void testAutoIdTemplate(id obj) {
  autoTemplateFunction<id, 2>(obj, obj, [Array new]); // no-warning
}

// rdar://12229679
@interface NSObject @end
typedef __builtin_va_list va_list;
@interface MyClass : NSObject
@end

@implementation MyClass
+ (void)fooMethod:(id)firstArg, ... {
    va_list args;

    __builtin_va_arg(args, id);
}
@end

namespace rdar12078752 {
  void f() {
    NSObject* o =0;
    __autoreleasing decltype(o) o2 = o;
    __autoreleasing auto o3 = o;
  }
}

namespace test_err_arc_array_param_no_ownership {
  template <class T>
  void func(T a) {}

  void test() {
    func([](A *a[]){}); // expected-error{{must explicitly describe intended ownership of an object array parameter}}
    func(^(A *a[]){}); // expected-error{{must explicitly describe intended ownership of an object array parameter}}
  }
}

namespace test_union {
  // Implicitly-declared special functions of a union are deleted by default if
  // ARC is enabled and the union has an ObjC pointer field.
  union U0 {
    id f0; // expected-note 7 {{'U0' is implicitly deleted because variant field 'f0' is an ObjC pointer}}
  };

  union U1 {
    __weak id f0; // expected-note 13 {{'U1' is implicitly deleted because variant field 'f0' is an ObjC pointer}}
    U1() = default; // expected-warning {{explicitly defaulted default constructor is implicitly deleted}} expected-note {{explicitly defaulted function was implicitly deleted here}}
    ~U1() = default; // expected-warning {{explicitly defaulted destructor is implicitly deleted}} expected-note 2{{explicitly defaulted function was implicitly deleted here}}
    U1(const U1 &) = default; // expected-warning {{explicitly defaulted copy constructor is implicitly deleted}} expected-note 2 {{explicitly defaulted function was implicitly deleted here}}
    U1(U1 &&) = default; // expected-warning {{explicitly defaulted move constructor is implicitly deleted}}
    U1 & operator=(const U1 &) = default; // expected-warning {{explicitly defaulted copy assignment operator is implicitly deleted}} expected-note 2 {{explicitly defaulted function was implicitly deleted here}}
    U1 & operator=(U1 &&) = default; // expected-warning {{explicitly defaulted move assignment operator is implicitly deleted}}
  };

  id getStrong();

  // If the ObjC pointer field of a union has a default member initializer, the
  // implicitly-declared default constructor of the union is not deleted by
  // default.
  union U2 {
    id f0 = getStrong(); // expected-note 4 {{'U2' is implicitly deleted because variant field 'f0' is an ObjC pointer}}
    ~U2();
  };

  // It's fine if the user has explicitly defined the special functions.
  union U3 {
    id f0;
    U3();
    ~U3();
    U3(const U3 &);
    U3(U3 &&);
    U3 & operator=(const U3 &);
    U3 & operator=(U3 &&);
  };

  // ObjC pointer fields in anonymous union fields delete the defaulted special
  // functions of the containing class.
  struct S0 {
    union {
      id f0; // expected-note 7 {{'' is implicitly deleted because variant field 'f0' is an ObjC pointer}}
      char f1;
    };
  };

  struct S1 {
    union {
      union { // expected-note 7 {{'S1' is implicitly deleted because field '' has a deleted}}
        id f0; // expected-note 3 {{'' is implicitly deleted because variant field 'f0' is an ObjC pointer}}
        char f1;
      };
      int f2;
    };
  };

  struct S2 {
    union {
      // FIXME: the note should say 'f0' is causing the special functions to be deleted.
      struct { // expected-note 7 {{'S2' is implicitly deleted because variant field '' has a non-trivial}}
        id f0;
        int f1;
      };
      int f2;
    };
    int f3;
  };

  U0 *x0;
  U1 *x1;
  U2 *x2;
  U3 *x3;
  S0 *x4;
  S1 *x5;
  S2 *x6;

  static union { // expected-error {{call to implicitly-deleted default constructor of}} expected-error {{attempt to use a deleted function}}
    id g0; // expected-note {{default constructor of '' is implicitly deleted because variant field 'g0' is an ObjC pointer}} \
           // expected-note {{destructor of '' is implicitly deleted because}}
  };

  static union { // expected-error {{call to implicitly-deleted default constructor of}} expected-error {{attempt to use a deleted function}}
    union { // expected-note {{default constructor of '' is implicitly deleted because field '' has a deleted default constructor}} \
            // expected-note {{destructor of '' is implicitly deleted because}}
      union { // expected-note {{default constructor of '' is implicitly deleted because field '' has a deleted default constructor}} \
              // expected-note {{destructor of '' is implicitly deleted because}}
        __weak id g1; // expected-note {{default constructor of '' is implicitly deleted because variant field 'g1' is an ObjC pointer}} \
                      // expected-note {{destructor of '' is implicitly deleted because}}
        int g2;
      };
      int g3;
    };
    int g4;
  };

  void testDefaultConstructor() {
    U0 t0; // expected-error {{call to implicitly-deleted default constructor}} expected-error {{attempt to use a deleted function}}
    U1 t1; // expected-error {{call to implicitly-deleted default constructor}} expected-error {{attempt to use a deleted function}}
    U2 t2;
    U3 t3;
    S0 t4; // expected-error {{call to implicitly-deleted default constructor}} expected-error {{attempt to use a deleted function}}
    S1 t5; // expected-error {{call to implicitly-deleted default constructor}} expected-error {{attempt to use a deleted function}}
    S2 t6; // expected-error {{call to implicitly-deleted default constructor}} expected-error {{attempt to use a deleted function}}
  }

  void testDestructor(U0 *u0, U1 *u1, U2 *u2, U3 *u3, S0 *s0, S1 *s1, S2 *s2) {
    delete u0; // expected-error {{attempt to use a deleted function}}
    delete u1; // expected-error {{attempt to use a deleted function}}
    delete u2;
    delete u3;
    delete s0; // expected-error {{attempt to use a deleted function}}
    delete s1; // expected-error {{attempt to use a deleted function}}
    delete s2; // expected-error {{attempt to use a deleted function}}
  }

  void testCopyConstructor(U0 *u0, U1 *u1, U2 *u2, U3 *u3, S0 *s0, S1 *s1, S2 *s2) {
    U0 t0(*u0); // expected-error {{call to implicitly-deleted copy constructor}}
    U1 t1(*u1); // expected-error {{call to implicitly-deleted copy constructor}}
    U2 t2(*u2); // expected-error {{call to implicitly-deleted copy constructor}}
    U3 t3(*u3);
    S0 t4(*s0); // expected-error {{call to implicitly-deleted copy constructor}}
    S1 t5(*s1); // expected-error {{call to implicitly-deleted copy constructor}}
    S2 t6(*s2); // expected-error {{call to implicitly-deleted copy constructor}}
  }

  void testCopyAssignment(U0 *u0, U1 *u1, U2 *u2, U3 *u3, S0 *s0, S1 *s1, S2 *s2) {
    *x0 = *u0; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x1 = *u1; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x2 = *u2; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x3 = *u3;
    *x4 = *s0; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x5 = *s1; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x6 = *s2; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
  }

  // The diagnostics below refer to the deleted copy constructors and assignment
  // operators since defaulted move constructors and assignment operators that are
  // defined as deleted are ignored by overload resolution.

  void testMoveConstructor(U0 *u0, U1 *u1, U2 *u2, U3 *u3, S0 *s0, S1 *s1, S2 *s2) {
    U0 t0(static_cast<U0 &&>(*u0)); // expected-error {{call to implicitly-deleted copy constructor}}
    U1 t1(static_cast<U1 &&>(*u1)); // expected-error {{call to implicitly-deleted copy constructor}}
    U2 t2(static_cast<U2 &&>(*u2)); // expected-error {{call to implicitly-deleted copy constructor}}
    U3 t3(static_cast<U3 &&>(*u3));
    S0 t4(static_cast<S0 &&>(*s0)); // expected-error {{call to implicitly-deleted copy constructor}}
    S1 t5(static_cast<S1 &&>(*s1)); // expected-error {{call to implicitly-deleted copy constructor}}
    S2 t6(static_cast<S2 &&>(*s2)); // expected-error {{call to implicitly-deleted copy constructor}}
  }

  void testMoveAssignment(U0 *u0, U1 *u1, U2 *u2, U3 *u3, S0 *s0, S1 *s1, S2 *s2) {
    *x0 = static_cast<U0 &&>(*u0); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x1 = static_cast<U1 &&>(*u1); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x2 = static_cast<U2 &&>(*u2); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x3 = static_cast<U3 &&>(*u3);
    *x4 = static_cast<S0 &&>(*s0); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x5 = static_cast<S1 &&>(*s1); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x6 = static_cast<S2 &&>(*s2); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
  }
}
