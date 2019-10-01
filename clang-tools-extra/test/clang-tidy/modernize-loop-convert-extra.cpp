// RUN: %check_clang_tidy %s modernize-loop-convert %t -- -- -I %S/Inputs/modernize-loop-convert

#include "structures.h"

namespace Dependency {

void f() {
  const int N = 6;
  const int M = 8;
  int Arr[N][M];

  for (int I = 0; I < N; ++I) {
    int A = 0;
    int B = Arr[I][A];
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Arr)
  // CHECK-FIXES-NEXT: int A = 0;
  // CHECK-FIXES-NEXT: int B = I[A];

  for (int J = 0; J < M; ++J) {
    int A = 0;
    int B = Arr[A][J];
  }
}

} // namespace Dependency

namespace NamingAlias {

const int N = 10;

Val Arr[N];
dependent<Val> V;
dependent<Val> *Pv;
Val &func(Val &);
void sideEffect(int);

void aliasing() {
  // If the loop container is only used for a declaration of a temporary
  // variable to hold each element, we can name the new variable for the
  // converted range-based loop as the temporary variable's name.

  // In the following case, "T" is used as a temporary variable to hold each
  // element, and thus we consider the name "T" aliased to the loop.
  // The extra blank braces are left as a placeholder for after the variable
  // declaration is deleted.
  for (int I = 0; I < N; ++I) {
    Val &T = Arr[I];
    {}
    int Y = T.X;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & T : Arr)
  // CHECK-FIXES-NOT: Val &{{[a-z_]+}} =
  // CHECK-FIXES-NEXT: {}
  // CHECK-FIXES-NEXT: int Y = T.X;

  // The container was not only used to initialize a temporary loop variable for
  // the container's elements, so we do not alias the new loop variable.
  for (int I = 0; I < N; ++I) {
    Val &T = Arr[I];
    int Y = T.X;
    int Z = Arr[I].X + T.X;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Arr)
  // CHECK-FIXES-NEXT: Val &T = I;
  // CHECK-FIXES-NEXT: int Y = T.X;
  // CHECK-FIXES-NEXT: int Z = I.X + T.X;

  for (int I = 0; I < N; ++I) {
    Val T = Arr[I];
    int Y = T.X;
    int Z = Arr[I].X + T.X;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Arr)
  // CHECK-FIXES-NEXT: Val T = I;
  // CHECK-FIXES-NEXT: int Y = T.X;
  // CHECK-FIXES-NEXT: int Z = I.X + T.X;

  // The same for pseudo-arrays like std::vector<T> (or here dependent<Val>)
  // which provide a subscript operator[].
  for (int I = 0; I < V.size(); ++I) {
    Val &T = V[I];
    {}
    int Y = T.X;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & T : V)
  // CHECK-FIXES-NEXT: {}
  // CHECK-FIXES-NEXT: int Y = T.X;

  // The same with a call to at()
  for (int I = 0; I < Pv->size(); ++I) {
    Val &T = Pv->at(I);
    {}
    int Y = T.X;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & T : *Pv)
  // CHECK-FIXES-NEXT: {}
  // CHECK-FIXES-NEXT: int Y = T.X;

  for (int I = 0; I < N; ++I) {
    Val &T = func(Arr[I]);
    int Y = T.X;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Arr)
  // CHECK-FIXES-NEXT: Val &T = func(I);
  // CHECK-FIXES-NEXT: int Y = T.X;

  int IntArr[N];
  for (unsigned I = 0; I < N; ++I) {
    if (int Alias = IntArr[I]) {
      sideEffect(Alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Alias : IntArr)
  // CHECK-FIXES-NEXT: if (Alias)

  for (unsigned I = 0; I < N; ++I) {
    while (int Alias = IntArr[I]) {
      sideEffect(Alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Alias : IntArr)
  // CHECK-FIXES-NEXT: while (Alias)

  for (unsigned I = 0; I < N; ++I) {
    switch (int Alias = IntArr[I]) {
    default:
      sideEffect(Alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-6]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Alias : IntArr)
  // CHECK-FIXES-NEXT: switch (Alias)

  for (unsigned I = 0; I < N; ++I) {
    for (int Alias = IntArr[I]; Alias < N; ++Alias) {
      sideEffect(Alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Alias : IntArr)
  // CHECK-FIXES-NEXT: for (; Alias < N; ++Alias)

  for (unsigned I = 0; I < N; ++I) {
    for (unsigned J = 0; int Alias = IntArr[I]; ++J) {
      sideEffect(Alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Alias : IntArr)
  // CHECK-FIXES-NEXT: for (unsigned J = 0; Alias; ++J)

  struct IntRef { IntRef(); IntRef(const int& i); operator int*(); };
  for (int I = 0; I < N; ++I) {
    IntRef Int(IntArr[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : IntArr)
  // CHECK-FIXES-NEXT: IntRef Int(I);

  int *PtrArr[N];
  for (unsigned I = 0; I < N; ++I) {
    const int* const P = PtrArr[I];
    printf("%d\n", *P);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto P : PtrArr)
  // CHECK-FIXES-NEXT: printf("%d\n", *P);

  IntRef Refs[N];
  for (unsigned I = 0; I < N; ++I) {
    int *P = Refs[I];
    printf("%d\n", *P);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Ref : Refs)
  // CHECK-FIXES-NEXT: int *P = Ref;
  // CHECK-FIXES-NEXT: printf("%d\n", *P);

  // Ensure that removing the alias doesn't leave empty lines behind.
  for (int I = 0; I < N; ++I) {
    auto &X = IntArr[I];
    X = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & X : IntArr) {
  // CHECK-FIXES-NEXT: {{^    X = 0;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
}

void refs_and_vals() {
  // The following tests check that the transform correctly preserves the
  // reference or value qualifiers of the aliased variable. That is, if the
  // variable was declared as a value, the loop variable will be declared as a
  // value and vice versa for references.

  S Ss;
  const S S_const = Ss;

  for (S::const_iterator It = S_const.begin(); It != S_const.end(); ++It) {
    MutableVal Alias = *It;
    {}
    Alias.X = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto Alias : S_const)
  // CHECK-FIXES-NOT: MutableVal {{[a-z_]+}} =
  // CHECK-FIXES-NEXT: {}
  // CHECK-FIXES-NEXT: Alias.X = 0;

  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    MutableVal Alias = *It;
    {}
    Alias.X = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto Alias : Ss)
  // CHECK-FIXES-NOT: MutableVal {{[a-z_]+}} =
  // CHECK-FIXES-NEXT: {}
  // CHECK-FIXES-NEXT: Alias.X = 0;

  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    MutableVal &Alias = *It;
    {}
    Alias.X = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Alias : Ss)
  // CHECK-FIXES-NOT: MutableVal &{{[a-z_]+}} =
  // CHECK-FIXES-NEXT: {}
  // CHECK-FIXES-NEXT: Alias.X = 0;

  dependent<int> Dep, Other;
  for (dependent<int>::iterator It = Dep.begin(), E = Dep.end(); It != E; ++It) {
    printf("%d\n", *It);
    const int& Idx = Other[0];
    unsigned Othersize = Other.size();
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Dep)
  // CHECK-FIXES-NEXT: printf("%d\n", It);
  // CHECK-FIXES-NEXT: const int& Idx = Other[0];
  // CHECK-FIXES-NEXT: unsigned Othersize = Other.size();

  for (int i = 0; i <  Other.size(); ++i) {
    Other.at(i);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & i : Other)
  // CHECK-FIXES: i;

  for (int I = 0, E = Dep.size(); I != E; ++I) {
    int Idx = Other.at(I);
    Other.at(I, I);  // Should not trigger assert failure.
  }
}

struct MemberNaming {
  const static int N = 10;
  int Ints[N], Ints_[N];
  dependent<int> DInts;
  void loops() {
    for (int I = 0; I < N; ++I) {
      printf("%d\n", Ints[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int Int : Ints)
    // CHECK-FIXES-NEXT: printf("%d\n", Int);

    for (int I = 0; I < N; ++I) {
      printf("%d\n", Ints_[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int Int : Ints_)
    // CHECK-FIXES-NEXT: printf("%d\n", Int);

    for (int I = 0; I < DInts.size(); ++I) {
      printf("%d\n", DInts[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int DInt : DInts)
    // CHECK-FIXES-NEXT: printf("%d\n", DInt);
  }

  void outOfLine();
};
void MemberNaming::outOfLine() {
  for (int I = 0; I < N; ++I) {
    printf("%d\n", Ints[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Int : Ints)
  // CHECK-FIXES-NEXT: printf("%d\n", Int);

  for (int I = 0; I < N; ++I) {
    printf("%d\n", Ints_[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Int : Ints_)
  // CHECK-FIXES-NEXT: printf("%d\n", Int);
}

} // namespace NamingAlias

namespace NamingConlict {

#define MAX(a, b) (a > b) ? a : b
#define DEF 5

const int N = 10;
int Nums[N];
int Sum = 0;

namespace ns {
struct St {
  int X;
};
}

void sameNames() {
  int Num = 0;
  for (int I = 0; I < N; ++I) {
    printf("Fibonacci number is %d\n", Nums[I]);
    Sum += Nums[I] + 2 + Num;
    (void)Nums[I];
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Nums)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2 + Num;
  // CHECK-FIXES-NEXT: (void)I;

  int Elem = 0;
  for (int I = 0; I < N; ++I) {
    printf("Fibonacci number is %d\n", Nums[I]);
    Sum += Nums[I] + 2 + Num + Elem;
    (void)Nums[I];
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Nums)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2 + Num + Elem;
  // CHECK-FIXES-NEXT: (void)I;
}

void oldIndexConflict() {
  for (int Num = 0; Num < N; ++Num) {
    printf("Num: %d\n", Nums[Num]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int Num : Nums)
  // CHECK-FIXES-NEXT: printf("Num: %d\n", Num);

  S Things;
  for (S::iterator Thing = Things.begin(), End = Things.end(); Thing != End; ++Thing) {
    printf("Thing: %d %d\n", Thing->X, (*Thing).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Thing : Things)
  // CHECK-FIXES-NEXT: printf("Thing: %d %d\n", Thing.X, Thing.X);
}

void macroConflict() {
  S MAXs;
  for (S::iterator It = MAXs.begin(), E = MAXs.end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
    printf("Max of 3 and 5: %d\n", MAX(3, 5));
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : MAXs)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);
  // CHECK-FIXES-NEXT: printf("Max of 3 and 5: %d\n", MAX(3, 5));

  for (S::const_iterator It = MAXs.begin(), E = MAXs.end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
    printf("Max of 3 and 5: %d\n", MAX(3, 5));
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto It : MAXs)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);
  // CHECK-FIXES-NEXT: printf("Max of 3 and 5: %d\n", MAX(3, 5));

  T DEFs;
  for (T::iterator It = DEFs.begin(), E = DEFs.end(); It != E; ++It) {
    if (*It == DEF) {
      printf("I found %d\n", *It);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : DEFs)
  // CHECK-FIXES-NEXT: if (It == DEF)
  // CHECK-FIXES-NEXT: printf("I found %d\n", It);
}

void keywordConflict() {
  T ints;
  for (T::iterator It = ints.begin(), E = ints.end(); It != E; ++It) {
    *It = 5;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : ints)
  // CHECK-FIXES-NEXT: It = 5;

  U __FUNCTION__s;
  for (U::iterator It = __FUNCTION__s.begin(), E = __FUNCTION__s.end();
       It != E; ++It) {
    int __FUNCTION__s_It = (*It).X + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : __FUNCTION__s)
  // CHECK-FIXES-NEXT: int __FUNCTION__s_It = It.X + 2;
}

void typeConflict() {
  T Vals;
  // Using the name "Val", although it is the name of an existing struct, is
  // safe in this loop since it will only exist within this scope.
  for (T::iterator It = Vals.begin(), E = Vals.end(); It != E; ++It)
    (void) *It;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & Val : Vals)

  // We cannot use the name "Val" in this loop since there is a reference to
  // it in the body of the loop.
  for (T::iterator It = Vals.begin(), E = Vals.end(); It != E; ++It) {
    *It = sizeof(Val);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Vals)
  // CHECK-FIXES-NEXT: It = sizeof(Val);

  typedef struct Val TD;
  U TDs;
  // Naming the variable "TD" within this loop is safe because the typedef
  // was never used within the loop.
  for (U::iterator It = TDs.begin(), E = TDs.end(); It != E; ++It)
    (void) *It;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & TD : TDs)

  // "TD" cannot be used in this loop since the typedef is being used.
  for (U::iterator It = TDs.begin(), E = TDs.end(); It != E; ++It) {
    TD V;
    V.X = 5;
    (void) *It;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : TDs)
  // CHECK-FIXES-NEXT: TD V;
  // CHECK-FIXES-NEXT: V.X = 5;

  using ns::St;
  T Sts;
  for (T::iterator It = Sts.begin(), E = Sts.end(); It != E; ++It) {
    *It = sizeof(St);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Sts)
  // CHECK-FIXES-NEXT: It = sizeof(St);
}

} // namespace NamingConflict

namespace FreeBeginEnd {

// FIXME: Loop Convert should detect free begin()/end() functions.

struct MyArray {
  unsigned size();
};

template <typename T>
struct MyContainer {
};

int *begin(const MyArray &Arr);
int *end(const MyArray &Arr);

template <typename T>
T *begin(const MyContainer<T> &C);
template <typename T>
T *end(const MyContainer<T> &C);

// The Loop Convert Transform doesn't detect free functions begin()/end() and
// so fails to transform these cases which it should.
void f() {
  MyArray Arr;
  for (unsigned I = 0, E = Arr.size(); I < E; ++I) {
  }

  MyContainer<int> C;
  for (int *I = begin(C), *E = end(C); I != E; ++I) {
  }
}

} // namespace FreeBeginEnd

namespace Nesting {

void g(S::iterator It);
void const_g(S::const_iterator It);
class Foo {
 public:
  void g(S::iterator It);
  void const_g(S::const_iterator It);
};

void f() {
  const int N = 10;
  const int M = 15;
  Val Arr[N];
  for (int I = 0; I < N; ++I) {
    for (int J = 0; J < N; ++J) {
      int K = Arr[I].X + Arr[J].X;
      // The repeat is there to allow FileCheck to make sure the two variable
      // names aren't the same.
      int L = Arr[I].X + Arr[J].X;
    }
  }
  // CHECK-MESSAGES: :[[@LINE-8]]:3: warning: use range-based for loop instead
  // CHECK-MESSAGES: :[[@LINE-8]]:5: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Arr)
  // CHECK-FIXES-NEXT: for (auto & J : Arr)
  // CHECK-FIXES-NEXT: int K = I.X + J.X;
  // CHECK-FIXES-NOT: int L = I.X + I.X;

  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. FIXME: update this test when that changes.
  Val Nest[N][M];
  for (int I = 0; I < N; ++I) {
    for (int J = 0; J < M; ++J) {
      printf("Got item %d", Nest[I][J].X);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Nest)
  // CHECK-FIXES-NEXT: for (int J = 0; J < M; ++J)
  // CHECK-FIXES-NEXT: printf("Got item %d", I[J].X);

  // Note that the order of M and N are switched for this test.
  for (int J = 0; J < M; ++J) {
    for (int I = 0; I < N; ++I) {
      printf("Got item %d", Nest[I][J].X);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use range-based for loop instead
  // CHECK-FIXES-NOT: for (auto & {{[a-zA-Z_]+}} : Nest[I])
  // CHECK-FIXES: for (int J = 0; J < M; ++J)
  // CHECK-FIXES-NEXT: for (auto & I : Nest)
  // CHECK-FIXES-NEXT: printf("Got item %d", I[J].X);

  // The inner loop is also convertible.
  Nested<T> NestT;
  for (Nested<T>::iterator I = NestT.begin(), E = NestT.end(); I != E; ++I) {
    for (T::iterator TI = (*I).begin(), TE = (*I).end(); TI != TE; ++TI) {
      printf("%d", *TI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : NestT)
  // CHECK-FIXES-NEXT: for (T::iterator TI = I.begin(), TE = I.end(); TI != TE; ++TI)
  // CHECK-FIXES-NEXT: printf("%d", *TI);

  // The inner loop is also convertible.
  Nested<S> NestS;
  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    for (S::const_iterator SI = (*I).begin(), SE = (*I).end(); SI != SE; ++SI) {
      printf("%d", *SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto I : NestS)
  // CHECK-FIXES-NEXT: for (S::const_iterator SI = I.begin(), SE = I.end(); SI != SE; ++SI)
  // CHECK-FIXES-NEXT: printf("%d", *SI);

  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    const S &Ss = *I;
    for (S::const_iterator SI = Ss.begin(), SE = Ss.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      const_g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto Ss : NestS)

  for (Nested<S>::iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    S &Ss = *I;
    for (S::iterator SI = Ss.begin(), SE = Ss.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Ss : NestS)

  Foo foo;
  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    const S &Ss = *I;
    for (S::const_iterator SI = Ss.begin(), SE = Ss.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      foo.const_g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto Ss : NestS)

  for (Nested<S>::iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    S &Ss = *I;
    for (S::iterator SI = Ss.begin(), SE = Ss.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      foo.g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Ss : NestS)

}

} // namespace Nesting

namespace SingleIterator {

void complexContainer() {
  X Exes[5];
  int Index = 0;

  for (S::iterator I = Exes[Index].getS().begin(), E = Exes[Index].getS().end(); I != E; ++I) {
    MutableVal K = *I;
    MutableVal J = *I;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Exes[Index].getS())
  // CHECK-FIXES-NEXT: MutableVal K = I;
  // CHECK-FIXES-NEXT: MutableVal J = I;
}

void f() {
  /// begin()/end() - based for loops here:
  T Tt;
  for (T::iterator It = Tt.begin(); It != Tt.end(); ++It) {
    printf("I found %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Tt)
  // CHECK-FIXES-NEXT: printf("I found %d\n", It);

  T *Pt;
  for (T::iterator It = Pt->begin(); It != Pt->end(); ++It) {
    printf("I found %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : *Pt)
  // CHECK-FIXES-NEXT: printf("I found %d\n", It);

  S Ss;
  for (S::iterator It = Ss.begin(); It != Ss.end(); ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  S *Ps;
  for (S::iterator It = Ps->begin(); It != Ps->end(); ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & P : *Ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", P.X);

  for (S::iterator It = Ss.begin(); It != Ss.end(); ++It) {
    printf("s has value %d\n", It->X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  for (S::iterator It = Ss.begin(); It != Ss.end(); ++It) {
    It->X = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: It.X = 3;

  for (S::iterator It = Ss.begin(); It != Ss.end(); ++It) {
    (*It).X = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: It.X = 3;

  for (S::iterator It = Ss.begin(); It != Ss.end(); ++It) {
    It->nonConstFun(4, 5);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: It.nonConstFun(4, 5);

  U Uu;
  for (U::iterator It = Uu.begin(); It != Uu.end(); ++It) {
    printf("s has value %d\n", It->X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Uu)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  for (U::iterator It = Uu.begin(); It != Uu.end(); ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Uu)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  U::iterator A;
  for (U::iterator I = Uu.begin(); I != Uu.end(); ++I)
    int K = A->X + I->X;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Uu)
  // CHECK-FIXES-NEXT: int K = A->X + I.X;

  dependent<int> V;
  for (dependent<int>::iterator It = V.begin();
       It != V.end(); ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);

  for (dependent<int>::iterator It(V.begin());
       It != V.end(); ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);

  doublyDependent<int, int> intmap;
  for (doublyDependent<int, int>::iterator It = intmap.begin();
       It != intmap.end(); ++It) {
    printf("intmap[%d] = %d", It->first, It->second);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : intmap)
  // CHECK-FIXES-NEXT: printf("intmap[%d] = %d", It.first, It.second);
}

void different_type() {
  // Tests to verify the proper use of auto where the init variable type and the
  // initializer type differ or are mostly the same except for const qualifiers.

  // Ss.begin() returns a type 'iterator' which is just a non-const pointer and
  // differs from const_iterator only on the const qualification.
  S Ss;
  for (S::const_iterator It = Ss.begin(); It != Ss.end(); ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  S *Ps;
  for (S::const_iterator It = Ps->begin(); It != Ps->end(); ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto P : *Ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", P.X);

  dependent<int> V;
  for (dependent<int>::const_iterator It = V.begin(); It != V.end(); ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);

  for (dependent<int>::const_iterator It(V.begin()); It != V.end(); ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);
}

} // namespace SingleIterator


namespace Macros {

#define TWO_PARAM(x, y) if (x == y) {}
#define THREE_PARAM(x, y, z) if (x == y) {z;}

const int N = 10;
int Arr[N];

void messing_with_macros() {
  for (int I = 0; I < N; ++I) {
    printf("Value: %d\n", Arr[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT:  printf("Value: %d\n", I);

  for (int I = 0; I < N; ++I) {
    printf("Value: %d\n", CONT Arr[I]);
  }

  // Multiple macro arguments.
  for (int I = 0; I < N; ++I) {
    TWO_PARAM(Arr[I], Arr[I]);
    THREE_PARAM(Arr[I], Arr[I], Arr[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: TWO_PARAM(I, I);
  // CHECK-FIXES-NEXT: THREE_PARAM(I, I, I);
}

} // namespace Macros

namespace Templates {

template <class Container>
void set_union(Container &container) {
  for (typename Container::const_iterator SI = container.begin(),
       SE = container.end(); SI != SE; ++SI) {
    (void) *SI;
  }

  S Ss;
  for (S::iterator SI = Ss.begin(), SE = Ss.end(); SI != SE; ++SI)
    (void) *SI;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & SI : Ss)
}

void template_instantiation() {
  S Ss;
  set_union(Ss);
}

} // namespace Templates

namespace Lambdas {

void capturesIndex() {
  const int N = 10;
  int Arr[N];
  // FIXME: the next four loops could be convertible, if the capture list is
  // also changed.

  for (int I = 0; I < N; ++I)
    auto F1 = [Arr, I]() { int R1 = Arr[I] + 1; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto F1 = [Arr, &I]() { int R1 = I + 1; };

  for (int I = 0; I < N; ++I)
    auto F2 = [Arr, &I]() { int R2 = Arr[I] + 3; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto F2 = [Arr, &I]() { int R2 = I + 3; };

  // FIXME: alias don't work if the index is captured.
  // Alias declared inside lambda (by value).
  for (int I = 0; I < N; ++I)
    auto F3 = [&Arr, I]() { int R3 = Arr[I]; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto F3 = [&Arr, &I]() { int R3 = I; };


  for (int I = 0; I < N; ++I)
    auto F4 = [&Arr, &I]() { int R4 = Arr[I]; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto F4 = [&Arr, &I]() { int R4 = I; };

  // Alias declared inside lambda (by reference).
  for (int I = 0; I < N; ++I)
    auto F5 = [&Arr, I]() { int &R5 = Arr[I]; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: auto F5 = [&Arr, &I]() { int &R5 = I; };


  for (int I = 0; I < N; ++I)
    auto F6 = [&Arr, &I]() { int &R6 = Arr[I]; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: auto F6 = [&Arr, &I]() { int &R6 = I; };

  for (int I = 0; I < N; ++I) {
    auto F = [Arr, I](int k) {
      printf("%d\n", Arr[I] + k);
    };
    F(Arr[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-6]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto F = [Arr, &I](int k)
  // CHECK-FIXES-NEXT: printf("%d\n", I + k);
  // CHECK-FIXES: F(I);
}

void implicitCapture() {
  const int N = 10;
  int Arr[N];
  // Index is used, not convertible.
  for (int I = 0; I < N; ++I) {
    auto G1 = [&]() {
      int R = Arr[I];
      int J = I;
    };
  }

  for (int I = 0; I < N; ++I) {
    auto G2 = [=]() {
      int R = Arr[I];
      int J = I;
    };
  }

  // Convertible.
  for (int I = 0; I < N; ++I) {
    auto G3 = [&]() {
      int R3 = Arr[I];
      int J3 = Arr[I] + R3;
    };
  }
  // CHECK-MESSAGES: :[[@LINE-6]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto G3 = [&]()
  // CHECK-FIXES-NEXT: int R3 = I;
  // CHECK-FIXES-NEXT: int J3 = I + R3;

  for (int I = 0; I < N; ++I) {
    auto G4 = [=]() {
      int R4 = Arr[I] + 5;
    };
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto G4 = [=]()
  // CHECK-FIXES-NEXT: int R4 = I + 5;

  // Alias by value.
  for (int I = 0; I < N; ++I) {
    auto G5 = [&]() {
      int R5 = Arr[I];
      int J5 = 8 + R5;
    };
  }
  // CHECK-MESSAGES: :[[@LINE-6]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int R5 : Arr)
  // CHECK-FIXES-NEXT: auto G5 = [&]()
  // CHECK-FIXES-NEXT: int J5 = 8 + R5;

  // Alias by reference.
  for (int I = 0; I < N; ++I) {
    auto G6 = [&]() {
      int &R6 = Arr[I];
      int J6 = -1 + R6;
    };
  }
  // CHECK-MESSAGES: :[[@LINE-6]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & R6 : Arr)
  // CHECK-FIXES-NEXT: auto G6 = [&]()
  // CHECK-FIXES-NEXT: int J6 = -1 + R6;
}

void iterators() {
  dependent<int> Dep;

  for (dependent<int>::iterator I = Dep.begin(), E = Dep.end(); I != E; ++I)
    auto H1 = [&I]() { int R = *I; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Dep)
  // CHECK-FIXES-NEXT: auto H1 = [&I]() { int R = I; };

  for (dependent<int>::iterator I = Dep.begin(), E = Dep.end(); I != E; ++I)
    auto H2 = [&]() { int R = *I + 2; };
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Dep)
  // CHECK-FIXES-NEXT: auto H2 = [&]() { int R = I + 2; };

  for (dependent<int>::const_iterator I = Dep.begin(), E = Dep.end();
       I != E; ++I)
    auto H3 = [I]() { int R = *I; };
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Dep)
  // CHECK-FIXES-NEXT: auto H3 = [&I]() { int R = I; };

  for (dependent<int>::const_iterator I = Dep.begin(), E = Dep.end();
       I != E; ++I)
    auto H4 = [&]() { int R = *I + 1; };
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Dep)
  // CHECK-FIXES-NEXT: auto H4 = [&]() { int R = I + 1; };

  for (dependent<int>::const_iterator I = Dep.begin(), E = Dep.end();
       I != E; ++I)
    auto H5 = [=]() { int R = *I; };
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int R : Dep)
  // CHECK-FIXES-NEXT: auto H5 = [=]() { };
}

void captureByValue() {
  // When the index is captured by value, we should replace this by a capture
  // by reference. This avoids extra copies.
  // FIXME: this could change semantics on array or pseudoarray loops if the
  // container is captured by copy.
  const int N = 10;
  int Arr[N];
  dependent<int> Dep;

  for (int I = 0; I < N; ++I) {
    auto C1 = [&Arr, I]() { if (Arr[I] == 1); };
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: auto C1 = [&Arr, &I]() { if (I == 1); };

  for (unsigned I = 0; I < Dep.size(); ++I) {
    auto C2 = [&Dep, I]() { if (Dep[I] == 2); };
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Dep)
  // CHECK-FIXES-NEXT: auto C2 = [&Dep, &I]() { if (I == 2); };
}

} // namespace Lambdas

namespace InitLists {

struct D { int Ii; };
struct E { D Dd; };
int g(int B);

void f() {
  const unsigned N = 3;
  int Array[N];

  // Subtrees of InitListExpr are visited twice. Test that we do not do repeated
  // replacements.
  for (unsigned I = 0; I < N; ++I) {
    int A{ Array[I] };
    int B{ g(Array[I]) };
    int C{ g( { Array[I] } ) };
    D Dd{ { g( { Array[I] } ) } };
    E Ee{ { { g( { Array[I] } ) } } };
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Array)
  // CHECK-FIXES-NEXT: int A{ I };
  // CHECK-FIXES-NEXT: int B{ g(I) };
  // CHECK-FIXES-NEXT: int C{ g( { I } ) };
  // CHECK-FIXES-NEXT: D Dd{ { g( { I } ) } };
  // CHECK-FIXES-NEXT: E Ee{ { { g( { I } ) } } };
}

} // namespace InitLists

void bug28341() {
  char v[5];
  for(int i = 0; i < 5; ++i) {
      unsigned char value = v[i];
      if (value > 127)
        ;
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for(unsigned char value : v)
  // CHECK-FIXES-NEXT: if (value > 127)
  }
}
