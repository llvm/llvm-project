template <typename T>
struct Generic {  T x; };

struct AbstractClass {
  virtual void pureMethod() = 0;
  virtual void aPureMethod(int (*fptr)(), Generic<int> y) = 0;
  virtual int anotherPureMethod(const int &x) const = 0;
  virtual int operator + (int) const = 0;
  virtual void otherMethod() { }
};

struct Base {
  virtual void nonAbstractClassMethod() { }
};

struct Target : Base, AbstractClass {
};
// CHECK1: "void pureMethod() override;\n\nvoid aPureMethod(int (*fptr)(), Generic<int> y) override;\n\nint anotherPureMethod(const int &x) const override;\n\nint operator+(int) const override;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1

// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:16:1 %s | FileCheck --check-prefix=CHECK1 %s

struct SubTarget : AbstractClass {
  int anotherPureMethod(const int &) const { return 0; }
#ifdef HAS_OP
  int operator + (int) const override { return 2; }
#endif
};

struct Target2 : SubTarget, Base {
};
// CHECK2: "void pureMethod() override;\n\nvoid aPureMethod(int (*fptr)(), Generic<int> y) override;\n\nint operator+(int) const override;\n\n" [[@LINE-1]]:1
// CHECK3: "void pureMethod() override;\n\nvoid aPureMethod(int (*fptr)(), Generic<int> y) override;\n\n" [[@LINE-2]]:1

// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:29:1 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:29:1 %s -DHAS_OP | FileCheck --check-prefix=CHECK3 %s

struct Abstract2 {
  virtual void firstMethod(int x, int y) = 0;
  virtual void secondMethod(int, int) { }
  virtual void thirdMethod(int a) = 0;
  virtual void fourthMethod() = 0;
};

struct FillInGoodLocations : Base, Abstract2 {

  void secondMethod(int, int) override; // comment

  void unrelatedMethod();

};
// CHECK4: "\n\nvoid firstMethod(int x, int y) override;\n\nvoid thirdMethod(int a) override;\n\nvoid fourthMethod() override;\n" [[@LINE-5]]:51 -> [[@LINE-5]]:51
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:44:1 %s | FileCheck --check-prefix=CHECK4 %s

struct FillInGoodLocations2 : FillInGoodLocations, AbstractClass {

  void fourthMethod() override;

  // comment
  void unrelatedMethod();

  int operator + (int) const override;
};
// CHECK5: "\n\nvoid firstMethod(int x, int y) override;\n\nvoid thirdMethod(int a) override;\n" [[@LINE-7]]:32 -> [[@LINE-7]]:32
// CHECK5-NEXT: "\n\nvoid pureMethod() override;\n\nvoid aPureMethod(int (*fptr)(), Generic<int> y) override;\n\nint anotherPureMethod(const int &x) const override;\n" [[@LINE-3]]:39 -> [[@LINE-3]]:39
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:54:1 %s | FileCheck --check-prefix=CHECK5 %s

struct FillInGoodLocations3 : Base, AbstractClass, Abstract2 {

  // comment
  void unrelatedMethod();

  void thirdMethod(int a) override;

  void firstMethod(int x, int y) override;

};
// CHECK6: "\n\nvoid fourthMethod() override;\n" [[@LINE-3]]:43 -> [[@LINE-3]]:43
// CHECK6-NEXT: "void pureMethod() override;\n\nvoid aPureMethod(int (*fptr)(), Generic<int> y) override;\n\nint anotherPureMethod(const int &x) const override;\n\nint operator+(int) const override;\n\n" [[@LINE-2]]:1 -> [[@LINE-2]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:67:1 %s | FileCheck --check-prefix=CHECK6 %s

struct FIllInGoodLocationsWithMacros : Abstract2 {
#define METHOD(decl) void decl override;

  METHOD(thirdMethod(int a))
  METHOD(firstMethod(int x, int y)) void foo();
};
// CHECK7: "\n\nvoid fourthMethod() override;\n" [[@LINE-2]]:36 -> [[@LINE-2]]:36
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:81:1 %s | FileCheck --check-prefix=CHECK7 %s

template<typename T>
class GenericType : Abstract2 {

};
// CHECK8: "void firstMethod(int x, int y) override;\n\nvoid thirdMethod(int a) override;\n\nvoid fourthMethod() override;\n\n" [[@LINE-1]]:1

struct GenericSubType : GenericType<int> {

};
// CHECK8: "void firstMethod(int x, int y) override;\n\nvoid thirdMethod(int a) override;\n\nvoid fourthMethod() override;\n\n" [[@LINE-1]]:1

// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:91:1 -at=%s:96:1 %s | FileCheck --check-prefix=CHECK8 %s


struct BaseClass2
{
    virtual ~BaseClass2();
    virtual int load() = 0;
};

// correct-implicit-destructor-placement: +1:1
struct DerivedImplicitDestructorClass2
: public BaseClass2
{

}; // CHECK-DESTRUCTOR: "int load() override;\n\n" [[@LINE]]:1

// Don't insert methods after the destructor:
// correct-destructor-placement: +1:1
struct DerivedExplicitDestructorClass2
: public BaseClass2 {
  ~DerivedImplicitDestructorClass2();


}; // CHECK-DESTRUCTOR: "int load() override;\n\n" [[@LINE]]:1

// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=correct-implicit-destructor-placement -at=correct-destructor-placement %s | FileCheck --check-prefix=CHECK-DESTRUCTOR %s
