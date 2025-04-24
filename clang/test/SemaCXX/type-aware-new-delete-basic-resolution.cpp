// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++26 -fexceptions    -fsized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++26 -fexceptions -fno-sized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++26 -fexceptions    -fsized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s        -std=c++26 -fexceptions -fno-sized-deallocation -fno-aligned-allocation

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

#if defined(__cpp_aligned_new)
#define ALLOCATION_ALIGNMENT , std::align_val_t
#else
#define ALLOCATION_ALIGNMENT
#endif

using size_t = __SIZE_TYPE__;

void *operator new(size_t);
void *operator new(size_t, std::align_val_t);
void operator delete(void *);

struct UntypedInclassNew {
  void *operator new(size_t) = delete; // #1
  void  operator delete(void *) = delete; //#2
};
void *operator new(std::type_identity<UntypedInclassNew>, size_t, std::align_val_t); // #3
void  operator delete(std::type_identity<UntypedInclassNew>, void*, size_t, std::align_val_t); // #4


struct __attribute__((aligned(128))) UntypedInclassNewOveraligned_NoAlignedAlloc {
  void *operator new(size_t) = delete; // #5
  void  operator delete(void *) = delete; // #6
};
void *operator new(std::type_identity<UntypedInclassNewOveraligned_NoAlignedAlloc>, size_t, std::align_val_t); // #7
void operator delete(std::type_identity<UntypedInclassNewOveraligned_NoAlignedAlloc>, void *, size_t, std::align_val_t); // #8

struct __attribute__((aligned(128))) UntypedInclassNewOveraligned_AlignedAlloc {
  void *operator new(size_t ALLOCATION_ALIGNMENT) = delete; // #9
  void  operator delete(void * ALLOCATION_ALIGNMENT) = delete; // #10
};
void *operator new(std::type_identity<UntypedInclassNewOveraligned_AlignedAlloc>, size_t, std::align_val_t); // #11
void  operator delete(std::type_identity<UntypedInclassNewOveraligned_AlignedAlloc>, void *, size_t, std::align_val_t); // #12

struct BasicClass {};
void *operator new(std::type_identity<BasicClass>, size_t, std::align_val_t) = delete; // #13
void  operator delete(std::type_identity<BasicClass>, void *, size_t, std::align_val_t) = delete; // #14

struct InclassNew1 {
  void *operator new(std::type_identity<InclassNew1>, size_t, std::align_val_t) = delete; // #15
  void  operator delete(std::type_identity<InclassNew1>, void *, size_t, std::align_val_t) = delete; // #16
};
void *operator new(std::type_identity<InclassNew1>, size_t, std::align_val_t); // #17
void  operator delete(std::type_identity<InclassNew1>, void *, size_t, std::align_val_t); // #18

struct InclassNew2 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t) = delete; // #19
  template <typename T> void  operator delete(std::type_identity<T>, void *, size_t, std::align_val_t) = delete; // #20
};
void *operator new(std::type_identity<InclassNew2>, size_t, std::align_val_t); // #21
void  operator delete(std::type_identity<InclassNew2>, void *, size_t, std::align_val_t); // #22

struct InclassNew3 {
  void *operator new(std::type_identity<InclassNew3>, size_t, std::align_val_t) = delete; // #23
  void  operator delete(std::type_identity<InclassNew3>, void*, size_t, std::align_val_t) = delete; // #24
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #25
  template <typename T> void  operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #26
};

struct __attribute__((aligned(128))) InclassNew4 {
  void *operator new(std::type_identity<InclassNew4>, size_t, std::align_val_t); // #27
  void  operator delete(std::type_identity<InclassNew4>, void*, size_t, std::align_val_t); // #28
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t) = delete; // #29
  template <typename T> void  operator delete(std::type_identity<T>, void *, size_t, std::align_val_t) = delete; // #30
};

struct InclassNew5 {
  InclassNew5();
  void *operator new(std::type_identity<InclassNew5>, size_t, std::align_val_t); // #31
  void  operator delete(void *); // #32
  void  operator delete(std::type_identity<InclassNew5>, void*, size_t, std::align_val_t) = delete; // #33
};

struct InclassNew6 {
  // expected-error@-1 {{declaration of type aware 'operator delete' in 'InclassNew6' must have matching type aware 'operator new'}}
  // expected-note@#36 {{unmatched type aware 'operator delete' declared here}}
  InclassNew6();
  void *operator new(size_t); // #34
  void  operator delete(void *) = delete; // #35
  void  operator delete(std::type_identity<InclassNew6>, void*, size_t, std::align_val_t) = delete; // #36
};

struct InclassNew7 {
  InclassNew7();
  void *operator new(std::type_identity<InclassNew7>, size_t, std::align_val_t); // #37
  void  operator delete(std::type_identity<InclassNew7>, void*, size_t, std::align_val_t); // #38
  void  operator delete(InclassNew7 *, std::destroying_delete_t) = delete; // #39
};

struct InclassNew8 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'InclassNew8' must have matching type aware 'operator delete'}}
  // expected-note@#40 {{unmatched type aware 'operator new' declared here}}
  InclassNew8();
  void *operator new(std::type_identity<InclassNew8>, size_t, std::align_val_t); // #40
  void operator delete(void*); // #41
};

struct InclassNew9 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'InclassNew9' must have matching type aware 'operator delete'}}
  // expected-note@#42 {{unmatched type aware 'operator new' declared here}}
  InclassNew9();
  void *operator new(std::type_identity<InclassNew9>, size_t, std::align_val_t); // #42
};

void operator delete(std::type_identity<InclassNew9>, void*, size_t, std::align_val_t); // #43

struct BaseClass1 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #44
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #45
  virtual ~BaseClass1();
};
BaseClass1::~BaseClass1() {
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#45 {{'operator delete<BaseClass1>' has been explicitly marked deleted here}}
}

struct SubClass1 : BaseClass1 { 
  virtual ~SubClass1();
};

SubClass1::~SubClass1() {
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#45 {{'operator delete<SubClass1>' has been explicitly marked deleted here}}
}

struct BaseClass2 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #46
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #47
  void operator delete(BaseClass2 *, std::destroying_delete_t);  // #48
  virtual ~BaseClass2();
};
BaseClass2::~BaseClass2(){
};

struct SubClass2 : BaseClass2 {
  SubClass2(); // Force exception cleanup which should invoke type aware delete
  virtual ~SubClass2();
};
SubClass2::~SubClass2(){
}

struct BaseClass3 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #49
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #50
  void operator delete(BaseClass3 *, std::destroying_delete_t) = delete; // #51
  virtual ~BaseClass3();
};
BaseClass3::~BaseClass3(){
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#51 {{'operator delete' has been explicitly marked deleted here}}
}

struct SubClass3 : BaseClass3 {
  virtual ~SubClass3();
};
SubClass3::~SubClass3(){
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#51 {{'operator delete' has been explicitly marked deleted here}}
}

template <typename A, typename B> concept Derived = requires (A * a, B *b) { a = b; };
template <typename A, typename B> concept Same = requires (std::type_identity<A> * a, std::type_identity<B> *b) { a = b; };

struct SubClass4;
struct BaseClass4 {
  template <Derived<SubClass4> T> void *operator new(std::type_identity<T>, size_t, std::align_val_t) = delete; // #52
  template <Derived<SubClass4> T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #53
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #54
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #55

  virtual ~BaseClass4();
};
BaseClass4::~BaseClass4() {
}

struct SubClass4 : BaseClass4 {
  virtual ~SubClass4();
};
SubClass4::~SubClass4(){
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#53 {{'operator delete<SubClass4>' has been explicitly marked deleted here}}
}

struct SubClass4_1 : SubClass4 {
  SubClass4_1();
};
struct SubClass4_2 : BaseClass4 {
};

struct SubClass5;
struct BaseClass5 {
  template <Same<SubClass5> T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #56
  template <Same<SubClass5> T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #57
  template <Derived<SubClass5> T> requires (!Same<SubClass5, T>) void *operator new(std::type_identity<T>, size_t, std::align_val_t) = delete; // #58
  template <Derived<SubClass5> T> requires (!Same<SubClass5, T>) void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #59
};

struct SubClass5 : BaseClass5 {
};
struct SubClass5_1 : SubClass5 {
};


struct BaseClass6 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #60
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #61
  BaseClass6();
  virtual ~BaseClass6();
};
BaseClass6::~BaseClass6(){
}

struct SubClass6_1 : BaseClass6 {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'SubClass6_1' must have matching type aware 'operator delete'}}
  // expected-note@#62 {{unmatched type aware 'operator new' declared here}}
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #62
  SubClass6_1();
};
struct SubClass6_2 : BaseClass6 {
  // expected-error@-1 {{declaration of type aware 'operator delete' in 'SubClass6_2' must have matching type aware 'operator new'}}
  // expected-note@#63 {{unmatched type aware 'operator delete' declared here}}
  template <typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #63
  SubClass6_2();
};

struct MultiDimensionArrayTest1 {
  int i;
  MultiDimensionArrayTest1();
  template <typename T, unsigned N> void *operator new[](std::type_identity<T[N]>, size_t, std::align_val_t) = delete; // #64
  template <typename T, unsigned N> void operator delete[](std::type_identity<T[N]>, void*, size_t, std::align_val_t) = delete; // #65
};

struct MultiDimensionArrayTest2 {
  int i;
  MultiDimensionArrayTest2();
  template <unsigned N> void *operator new[](std::type_identity<MultiDimensionArrayTest2[N]>, size_t, std::align_val_t) = delete; // #66
  template <unsigned N> void operator delete[](std::type_identity<MultiDimensionArrayTest2[N]>, void*, size_t, std::align_val_t) = delete; // #67
};

struct MultiDimensionArrayTest3 {
  int i;
  MultiDimensionArrayTest3();
  template <unsigned N> requires (N%4 == 0) void *operator new[](std::type_identity<MultiDimensionArrayTest3[N]>, size_t, std::align_val_t) = delete; // #68
  template <unsigned N> requires (N%4 == 0) void operator delete[](std::type_identity<MultiDimensionArrayTest3[N]>, void*, size_t, std::align_val_t) = delete; // #69
};

struct ClassScopedTemplatePackStruct {
  template <class T, class... Pack> void *operator new(std::type_identity<T>, size_t, std::align_val_t, Pack...);
  template <class T, class... Pack> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t, Pack...); // #70
};

void test() {
  
  // untyped in class declaration wins
  UntypedInclassNew *O1 = new UntypedInclassNew;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#1 {{candidate function has been explicitly deleted}}
  delete O1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#2 {{'operator delete' has been explicitly marked deleted here}}

  // untyped in class declaration wins, even though global is aligned and in class is not
  UntypedInclassNewOveraligned_NoAlignedAlloc *O2 = new UntypedInclassNewOveraligned_NoAlignedAlloc;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#5 {{candidate function has been explicitly deleted}}
  delete O2;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#6 {{'operator delete' has been explicitly marked deleted here}}

  // untyped in class declaration wins
  UntypedInclassNewOveraligned_AlignedAlloc *O3 = new UntypedInclassNewOveraligned_AlignedAlloc;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#9 {{candidate function has been explicitly deleted}}
  delete O3;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#10 {{'operator delete' has been explicitly marked deleted here}}

  // We resolve the explicitly typed free operator
  BasicClass *O4 = new BasicClass;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#13 {{candidate function has been explicitly deleted}}
  // expected-note@#3 {{candidate function not viable: no known conversion from 'type_identity<BasicClass>' to 'type_identity<UntypedInclassNew>' for 1st argument}}
  // expected-note@#17 {{candidate function not viable: no known conversion from 'type_identity<BasicClass>' to 'type_identity<InclassNew1>' for 1st argument}}
  // expected-note@#21 {{candidate function not viable: no known conversion from 'type_identity<BasicClass>' to 'type_identity<InclassNew2>' for 1st argument}}
  // expected-note@#7 {{candidate function not viable: no known conversion from 'type_identity<BasicClass>' to 'type_identity<UntypedInclassNewOveraligned_NoAlignedAlloc>' for 1st argument}}
  // expected-note@#11 {{candidate function not viable: no known conversion from 'type_identity<BasicClass>' to 'type_identity<UntypedInclassNewOveraligned_AlignedAlloc>' for 1st argument}}

  delete O4;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#14 {{'operator delete' has been explicitly marked deleted here}}

  // We resolve the explicitly typed in class operator
  InclassNew1 *O5 = new InclassNew1;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#15 {{candidate function has been explicitly deleted}}
  delete O5;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#16 {{'operator delete' has been explicitly marked deleted here}}

  // We resolve the unconstrained in class operators over the constrained free operators
  InclassNew2 *O6 = new InclassNew2;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#19 {{candidate function [with T = InclassNew2] has been explicitly deleted}}
  delete O6;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#20 {{'operator delete<InclassNew2>' has been explicitly marked deleted here}}

  // We prefer the constrained in class operators over the unconstrained variants
  InclassNew3 *O7 = new InclassNew3;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#23 {{candidate function has been explicitly deleted}}
  // expected-note@#25 {{candidate function [with T = InclassNew3]}}
  delete O7;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#24 {{'operator delete' has been explicitly marked deleted here}}

  // Constructor clean up invokes typed operator if typed new was used
  InclassNew5 *O9 = new InclassNew5;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#33 {{'operator delete' has been explicitly marked deleted here}}
  delete O9;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#33 {{'operator delete' has been explicitly marked deleted here}}

  // Constructor clean up invokes untyped delete if untyped delete was used
  InclassNew6 *O10 = new InclassNew6;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#36 {{'operator delete' has been explicitly marked deleted here}}
  // expected-error@-3 {{type aware 'operator delete' requires a matching type aware 'operator new' to be declared in the same scope}}
  // expected-note@#34 {{non-type aware 'operator new' declared here in 'InclassNew6'}}
  // expected-note@#36 {{type aware 'operator delete' declared here}}
  delete O10;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#36 {{'operator delete' has been explicitly marked deleted here}}

  // Destroying delete is prefered over typed delete
  InclassNew7 *O11 = new InclassNew7;
  delete O11;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#39 {{'operator delete' has been explicitly marked deleted here}}

  InclassNew8 *O12 = new InclassNew8;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#40 {{type aware 'operator new' declared here in 'InclassNew8'}}
  // expected-note@#41 {{non-type aware 'operator delete' declared here}}
  delete O12;

  InclassNew9 *O13 = new InclassNew9;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#42 {{type aware 'operator new' declared here in 'InclassNew9'}}

  delete O13;

  // Creating the virtual destructor for an type requires the deleting destructor
  // for that type
  SubClass1 *O14 = new SubClass1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#45 {{'operator delete<SubClass1>' has been explicitly marked deleted here}}

  delete O14;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#45 {{'operator delete<SubClass1>' has been explicitly marked deleted here}}

  SubClass2 *O15 = new SubClass2;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#47 {{'operator delete<SubClass2>' has been explicitly marked deleted here}}
  delete O15;

  // Deletion triggers destroying delete despite type aware delete
  SubClass3 *O16 = new SubClass3;
  delete O16;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#51 {{'operator delete' has been explicitly marked deleted here}}

  SubClass4 *O17 = new SubClass4;
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#52 {{candidate function [with T = SubClass4] has been explicitly deleted}}
  // expected-note@#54 {{candidate function [with T = SubClass4]}}
  delete O17;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#53 {{'operator delete<SubClass4>' has been explicitly marked deleted here}}

  SubClass4_1 *O18 = new SubClass4_1;
  delete O18;

  SubClass4_2 *O19 = new SubClass4_2;
  delete O19;

  SubClass5 *O20 = new SubClass5;
  delete O20;

  SubClass5_1 *O21 = new SubClass5_1;
  // expected-error@-1 {{no matching function for call to 'operator new'}}
  delete O21;
  // expected-error@-1 {{no suitable member 'operator delete' in 'SubClass5_1'}}
  // expected-note@#57 {{member 'operator delete' declared here}}
  // expected-note@#59 {{member 'operator delete' declared here}}

  SubClass6_1 *O22 = new SubClass6_1;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#62 {{type aware 'operator new' declared here in 'SubClass6_1'}}
  // expected-note@#61 {{type aware 'operator delete' declared here in 'BaseClass6'}}
  delete O22;

  SubClass6_2 *O23 = new SubClass6_2;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware 'operator delete' to be declared in the same scope}}
  // expected-note@#60 {{type aware 'operator new' declared here in 'BaseClass6'}}
  // expected-note@#63 {{type aware 'operator delete' declared here in 'SubClass6_2'}}
  delete O23;

  MultiDimensionArrayTest1 *O24 = new MultiDimensionArrayTest1;
  delete O24;

  MultiDimensionArrayTest1 *O25 = new MultiDimensionArrayTest1[10];
  // expected-error@-1 {{no matching function for call to 'operator new[]'}}
  delete [] O25;
  // expected-error@-1 {{no suitable member 'operator delete[]' in 'MultiDimensionArrayTest1'}}
  // expected-note@#65 {{member 'operator delete[]' declared here}}

  {
    using InnerArray = MultiDimensionArrayTest1[3];
    InnerArray *O26 = new InnerArray[7];
    // expected-error@-1 {{call to deleted function 'operator new[]'}}
    // expected-note@#64 {{candidate function [with T = MultiDimensionArrayTest1, N = 3] has been explicitly deleted}}
    delete [] O26;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#65 {{'operator delete[]<MultiDimensionArrayTest1, 3U>' has been explicitly marked deleted here}}
  }
  {
    using InnerArray = MultiDimensionArrayTest2[3];
    InnerArray *O27 = new InnerArray[7];
    // expected-error@-1 {{call to deleted function 'operator new[]'}}
    // expected-note@#66 {{candidate function [with N = 3] has been explicitly deleted}}
    delete [] O27;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#67 {{'operator delete[]<3U>' has been explicitly marked deleted here}}
  }
  {
    using InnerArray = MultiDimensionArrayTest3[3];
    InnerArray *O28 = new InnerArray[3];
    // expected-error@-1 {{no matching function for call to 'operator new[]'}}
    delete [] O28;
    // expected-error@-1 {{no suitable member 'operator delete[]' in 'MultiDimensionArrayTest3'}}
    // expected-note@#69 {{member 'operator delete[]' declared here}}
  }
  {
    using InnerArray = MultiDimensionArrayTest3[4];
    InnerArray *O29 = new InnerArray[3];
    // expected-error@-1 {{call to deleted function 'operator new[]'}}
    // expected-note@#68 {{candidate function [with N = 4] has been explicitly deleted}}
    delete [] O29;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#69 {{'operator delete[]<4U>' has been explicitly marked deleted here}}
  }
  {
    ClassScopedTemplatePackStruct *O30 = new ClassScopedTemplatePackStruct;
    delete O30;
    // expected-error@-1 {{no suitable member 'operator delete' in 'ClassScopedTemplatePackStruct'}}
    // expected-note@#70 {{member 'operator delete' declared here}}
  }
}
