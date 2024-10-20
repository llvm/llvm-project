// RUN: %clang_cc1 -fsyntax-only -verify %s        -std=c++2c -fexperimental-cxx-type-aware-allocators -fexceptions
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTADD -std=c++2c -fexperimental-cxx-type-aware-allocators -fcxx-type-aware-destroying-delete -fexceptions

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

static_assert(__has_feature(cxx_type_aware_allocators));
#ifdef TADD
static_assert(__has_feature(cxx_type_aware_destroying_delete));
#else
static_assert(!__has_feature(cxx_type_aware_destroying_delete));
#endif

using size_t = __SIZE_TYPE__;

void *operator new(size_t);
void *operator new(size_t, std::align_val_t);
void operator delete(void *);

struct UntypedInclassNew {
  void *operator new(size_t) = delete; // expected-note {{candidate function has been explicitly deleted}}
  void  operator delete(void *) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}
};
void *operator new(std::type_identity<UntypedInclassNew>, size_t); // expected-note {{candidate function not viable}}
void  operator delete(std::type_identity<UntypedInclassNew>, void*);

struct __attribute__((aligned(128))) UntypedInclassNewOveraligned_NoAlignedAlloc {
  void *operator new(size_t) = delete; // expected-note {{candidate function has been explicitly deleted}}
  void  operator delete(void *) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}
};
void *operator new(std::type_identity<UntypedInclassNewOveraligned_NoAlignedAlloc>, size_t, std::align_val_t); // expected-note {{candidate function not viable}}
void operator delete(std::type_identity<UntypedInclassNewOveraligned_NoAlignedAlloc>, void *, std::align_val_t);

struct __attribute__((aligned(128))) UntypedInclassNewOveraligned_AlignedAlloc {
  void *operator new(size_t, std::align_val_t) = delete; // expected-note {{candidate function has been explicitly deleted}}
  void  operator delete(void *, std::align_val_t) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}
};
void *operator new(std::type_identity<UntypedInclassNewOveraligned_AlignedAlloc>, size_t, std::align_val_t); // expected-note {{candidate function not viable}}
void  operator delete(std::type_identity<UntypedInclassNewOveraligned_AlignedAlloc>, void *, std::align_val_t);

struct BasicClass {};
void *operator new(std::type_identity<BasicClass>, size_t) = delete; // expected-note {{candidate function has been explicitly deleted}}
void  operator delete(std::type_identity<BasicClass>, void *) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}

struct InclassNew1 {
  void *operator new(std::type_identity<InclassNew1>, size_t) = delete; // expected-note {{candidate function has been explicitly deleted}}
  void  operator delete(std::type_identity<InclassNew1>, void *) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}
};
void *operator new(std::type_identity<InclassNew1>, size_t); // expected-note {{candidate function not viable}}
void  operator delete(std::type_identity<InclassNew1>, void *);

struct InclassNew2 {
  template <typename T> void *operator new(std::type_identity<T>, size_t) = delete; // expected-note {{candidate function [with T = InclassNew2] has been explicitly deleted}}
  template <typename T> void  operator delete(std::type_identity<T>, void *) = delete; // expected-note {{'operator delete<InclassNew2>' has been explicitly marked deleted here}}
};
void *operator new(std::type_identity<InclassNew2>, size_t); // expected-note {{candidate function not viable}}
void  operator delete(std::type_identity<InclassNew2>, void *);

struct InclassNew3 {
  void *operator new(std::type_identity<InclassNew3>, size_t) = delete; // expected-note {{candidate function has been explicitly deleted}}
  void  operator delete(std::type_identity<InclassNew3>, void *) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}
  template <typename T> void *operator new(std::type_identity<T>, size_t); // expected-note {{candidate function [with T = InclassNew3]}}
  template <typename T> void  operator delete(std::type_identity<T>, void *);
};

struct __attribute__((aligned(128))) InclassNew4 {
  void *operator new(std::type_identity<InclassNew4>, size_t); // expected-note {{candidate function not viable}}
  void  operator delete(std::type_identity<InclassNew4>, void *);
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t) = delete; // expected-note {{candidate function [with T = InclassNew4] has been explicitly deleted}}
  template <typename T> void  operator delete(std::type_identity<T>, void *, std::align_val_t) = delete; // expected-note {{'operator delete<InclassNew4>' has been explicitly marked deleted here}}
};

struct InclassNew5 {
  InclassNew5();
  void *operator new(std::type_identity<InclassNew5>, size_t);
  void  operator delete(void *);
  void  operator delete(std::type_identity<InclassNew5>, void *) = delete; // expected-note 2 {{'operator delete' has been explicitly marked deleted here}}
};

struct InclassNew6 {
  InclassNew6();
  void *operator new(size_t); // expected-note {{non-type aware 'operator new' declared here}}
  void  operator delete(void *) = delete;
  void  operator delete(std::type_identity<InclassNew6>, void *) = delete; // expected-note 2 {{'operator delete' has been explicitly marked deleted here}}
  // expected-note@-1 {{type aware 'operator delete' declared here}}
};

struct InclassNew7 {
  InclassNew7();
  void *operator new(std::type_identity<InclassNew7>, size_t);
  void  operator delete(std::type_identity<InclassNew7>, void *);
  void  operator delete(InclassNew7 *, std::destroying_delete_t) = delete; // expected-note {{'operator delete' has been explicitly marked deleted here}}
};

struct InclassNew8 {
  InclassNew8();
  void *operator new(std::type_identity<InclassNew8>, size_t);  // expected-note {{type aware 'operator new' declared here}}
  void operator delete(void*);  // expected-note {{non-type aware 'operator delete' declared here}}
};

struct InclassNew9 {
  InclassNew9();
  void *operator new(std::type_identity<InclassNew9>, size_t);
  // expected-note@-1 {{type aware 'operator new' found in 'InclassNew9'}}
};

void operator delete(std::type_identity<InclassNew9>, void*);
// expected-note@-1 {{type aware 'operator delete' found in the global namespace}}

struct BaseClass1 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  template <typename T> void operator delete(std::type_identity<T>, void*) = delete;
  // expected-note@-1 2 {{'operator delete<SubClass1>' has been explicitly marked deleted here}}
  virtual ~BaseClass1(); // expected-note {{overridden virtual function is here}}
};

struct SubClass1 : BaseClass1 { 
  // expected-error@-1 {{deleted function '~SubClass1' cannot override a non-deleted function}}
  // expected-note@-2 {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
};

struct BaseClass2 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  template <typename T> void operator delete(std::type_identity<T>, void*) = delete;
  // expected-note@-1 {{'operator delete<SubClass2>' has been explicitly marked deleted here}}
  void operator delete(BaseClass2 *, std::destroying_delete_t); 
  virtual ~BaseClass2();
};

struct SubClass2 : BaseClass2 {
  SubClass2(); // Force exception cleanup which should invoke type aware delete
};

struct BaseClass3 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  template <typename T> void operator delete(std::type_identity<T>, void*);
  void operator delete(BaseClass3 *, std::destroying_delete_t) = delete;  // expected-note {{'operator delete' has been explicitly marked deleted here}}
  virtual ~BaseClass3(); // expected-note {{overridden virtual function is here}}
};
struct SubClass3 : BaseClass3 {
  // expected-error@-1 {{deleted function '~SubClass3' cannot override a non-deleted function}}
  // expected-note@-2 {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
};

template <typename A, typename B> concept Derived = requires (A * a, B *b) { a = b; };
template <typename A, typename B> concept Same = requires (std::type_identity<A> * a, std::type_identity<B> *b) { a = b; };

struct SubClass4;
struct BaseClass4 {
  template <Derived<SubClass4> T> void *operator new(std::type_identity<T>, size_t) = delete;
  // expected-note@-1 {{candidate function [with T = SubClass4] has been explicitly deleted}}
  template <Derived<SubClass4> T> void operator delete(std::type_identity<T>, void*) = delete;
  // expected-note@-1 {{'operator delete<SubClass4>' has been explicitly marked deleted here}}
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  // expected-note@-1 {{candidate function [with T = SubClass4]}}
  template <typename T> void operator delete(std::type_identity<T>, void*);

  virtual ~BaseClass4(); // expected-note {{overridden virtual function is here}}
};

struct SubClass4 : BaseClass4 {
  // expected-error@-1 {{deleted function '~SubClass4' cannot override a non-deleted function}}
  // expected-note@-2 2 {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
};
struct SubClass4_1 : SubClass4 {
  // expected-note@-1 {{destructor of 'SubClass4_1' is implicitly deleted because base class 'SubClass4' has a deleted destructor}}
  SubClass4_1();
};
struct SubClass4_2 : BaseClass4 {
};

struct SubClass5;
struct BaseClass5 {
  template <Same<SubClass5> T> void *operator new(std::type_identity<T>, size_t);
  template <Same<SubClass5> T> void operator delete(std::type_identity<T>, void*); // expected-note {{member 'operator delete' declared here}}
  template <Derived<SubClass5> T> requires (!Same<SubClass5, T>) void *operator new(std::type_identity<T>, size_t) = delete;
  template <Derived<SubClass5> T> requires (!Same<SubClass5, T>) void operator delete(std::type_identity<T>, void*) = delete;
  // expected-note@-1 {{member 'operator delete' declared here}}
};

struct SubClass5 : BaseClass5 {
};
struct SubClass5_1 : SubClass5 {
};


struct BaseClass6 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  // expected-note@-1 {{type aware 'operator new' found in 'BaseClass6'}}
  template <typename T> void operator delete(std::type_identity<T>, void*);
  // expected-note@-1 {{type aware 'operator delete' found in 'BaseClass6'}}
  BaseClass6();
  virtual ~BaseClass6();
};

struct SubClass6_1 : BaseClass6 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  // expected-note@-1 {{type aware 'operator new' found in 'SubClass6_1'}}
  SubClass6_1();
};
struct SubClass6_2 : BaseClass6 {
  template <typename T> void operator delete(std::type_identity<T>, void*);
  // expected-note@-1 {{type aware 'operator delete' found in 'SubClass6_2'}}
  SubClass6_2();
};


void test() {
  
  // untyped in class declaration wins
  UntypedInclassNew *O1 = new UntypedInclassNew; // expected-error {{call to deleted function 'operator new'}}
  delete O1; // expected-error {{attempt to use a deleted function}}

  // untyped in class declaration wins, even though global is aligned and in class is not
  UntypedInclassNewOveraligned_NoAlignedAlloc *O2 = new UntypedInclassNewOveraligned_NoAlignedAlloc; // expected-error {{call to deleted function 'operator new'}}
  delete O2; // expected-error {{attempt to use a deleted function}}

  // untyped in class declaration wins
  UntypedInclassNewOveraligned_AlignedAlloc *O3 = new UntypedInclassNewOveraligned_AlignedAlloc; // expected-error {{call to deleted function 'operator new'}}
  delete O3; // expected-error {{attempt to use a deleted function}}

  // We resolve the explicitly typed free operator
  BasicClass *O4 = new BasicClass; // expected-error {{call to deleted function 'operator new'}}
  delete O4; // expected-error {{attempt to use a deleted function}}

  // We resolve the explicitly typed in class operator
  InclassNew1 *O5 = new InclassNew1; // expected-error {{call to deleted function 'operator new'}}
  delete O5; // expected-error {{attempt to use a deleted function}}

  // We resolve the unconstrained in class operators over the constrained free operators
  InclassNew2 *O6 = new InclassNew2; // expected-error {{call to deleted function 'operator new'}}
  delete O6; // expected-error {{attempt to use a deleted function}}

  // We prefer the constrained in class operators over the unconstrained variants
  InclassNew3 *O7 = new InclassNew3; // expected-error {{call to deleted function 'operator new'}}
  delete O7; // expected-error {{attempt to use a deleted function}}

  // We prefer the aligned but unconstrained operators over the unaligned but constrained variants
  InclassNew4 *O8 = new InclassNew4; // expected-error {{call to deleted function 'operator new'}}
  delete O8; // expected-error {{attempt to use a deleted function}}

  // Constructor clean up invokes typed operator if typed new was used
  InclassNew5 *O9 = new InclassNew5; // expected-error {{attempt to use a deleted function}}
  delete O9; // expected-error {{attempt to use a deleted function}}

  // Are these reasonable? Should we ensure that declaration of new vs delete have consistent type
  // semantics? How do we define consistent?
  // Constructor clean up invokes untyped delete if untyped delete was used
  InclassNew6 *O10 = new InclassNew6; // expected-error {{attempt to use a deleted function}}
  // expected-warning@-1 {{mismatching type aware allocation operators for constructor cleanup}}
  delete O10; //expected-error {{attempt to use a deleted function}}

  // Destroying delete is prefered over typed delete
  InclassNew7 *O11 = new InclassNew7;
  delete O11; // expected-error {{attempt to use a deleted function}}

  InclassNew8 *O12 = new InclassNew8;
  // expected-warning@-1 {{mismatching type aware allocation operators for constructor cleanup}}
  delete O12;

  InclassNew9 *O13 = new InclassNew9;
  // expected-error@-1 {{type aware 'operator new' requires matching 'operator delete' in 'InclassNew9'}}
  delete O13;

  // Creating the virtual destructor for an type requires the deleting destructor
  // for that type
  SubClass1 *O14 = new SubClass1; // expected-error {{attempt to use a deleted function}}
  delete O14; // expected-error {{attempt to use a deleted function}}

  SubClass2 *O15 = new SubClass2; // expected-error {{attempt to use a deleted function}}
  delete O15; 

  // Deletion triggers destroying delete despite type aware delete
  SubClass3 *O16 = new SubClass3;
  delete O16; // expected-error {{attempt to use a deleted function}}

  SubClass4 *O17 = new SubClass4; // expected-error {{call to deleted function 'operator new'}}
  delete O17; // expected-error {{attempt to use a deleted function}}

  SubClass4_1 *O18 = new SubClass4_1;
  delete O18; // expected-error {{attempt to use a deleted function}}

  SubClass4_2 *O19 = new SubClass4_2;
  delete O19;

  SubClass5 *O20 = new SubClass5;
  delete O20;

  SubClass5_1 *O21 = new SubClass5_1; // expected-error {{no matching function for call to 'operator new'}}
  delete O21; // expected-error {{no suitable member 'operator delete' in 'SubClass5_1'}}

  SubClass6_1 *O22 = new SubClass6_1;
  // expected-error@-1 {{type aware 'operator new' requires matching 'operator delete' in 'SubClass6_1'}}
  delete O22;

  SubClass6_2 *O23 = new SubClass6_2;
  // expected-error@-1 {{type aware 'operator new' requires matching 'operator delete' in 'BaseClass6'}}
  delete O23;
}
