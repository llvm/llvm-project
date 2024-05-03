// RUN: %check_clang_tidy %s modernize-loop-convert %t -- -- -I %S/Inputs/loop-convert

#include "structures.h"

namespace Array {

const int N = 6;
const int NMinusOne = N - 1;
int Arr[N] = {1, 2, 3, 4, 5, 6};
const int ConstArr[N] = {1, 2, 3, 4, 5, 6};
int (*PArr)[N] = &Arr;

void f() {
  int Sum = 0;

  for (int I = 0; I < N; ++I) {
    Sum += Arr[I];
    int K;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: Sum += I;
  // CHECK-FIXES-NEXT: int K;

  for (int I = 0; I < N; ++I) {
    printf("Fibonacci number is %d\n", Arr[I]);
    Sum += Arr[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0; I < N; ++I) {
    int X = Arr[I];
    int Y = Arr[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: int X = I;
  // CHECK-FIXES-NEXT: int Y = I + 2;

  for (int I = 0; I < N; ++I) {
    int X = N;
    X = Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: int X = N;
  // CHECK-FIXES-NEXT: X = I;

  for (int I = 0; I < N; ++I) {
    Arr[I] += 1;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: I += 1;

  for (int I = 0; I < N; ++I) {
    int X = Arr[I] + 2;
    Arr[I]++;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: int X = I + 2;
  // CHECK-FIXES-NEXT: I++;

  for (int I = 0; I < N; ++I) {
    Arr[I] = 4 + Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: I = 4 + I;

  for (int I = 0; I < NMinusOne + 1; ++I) {
    Sum += Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : Arr)
  // CHECK-FIXES-NEXT: Sum += I;

  for (int I = 0; I < N; ++I) {
    printf("Fibonacci number %d has address %p\n", Arr[I], &Arr[I]);
    Sum += Arr[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number %d has address %p\n", I, &I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  int Matrix[N][12];
  unsigned size = 0;
  for (int I = 0; I < N; ++I) {
      size += sizeof(Matrix[I]);
      size += sizeof Matrix[I];
      size += sizeof((Matrix[I]));
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Matrix)
  // CHECK-FIXES-NEXT: size += sizeof(I);
  // CHECK-FIXES-NEXT: size += sizeof I;
  // CHECK-FIXES-NEXT: size += sizeof(I);

  Val Teas[N];
  for (int I = 0; I < N; ++I) {
    Teas[I].g();
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Tea : Teas)
  // CHECK-FIXES-NEXT: Tea.g();

  for (int I = 0; N > I; ++I) {
    printf("Fibonacci number %d has address %p\n", Arr[I], &Arr[I]);
    Sum += Arr[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number %d has address %p\n", I, &I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0; N != I; ++I) {
    printf("Fibonacci number %d has address %p\n", Arr[I], &Arr[I]);
    Sum += Arr[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number %d has address %p\n", I, &I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0; I != N; ++I) {
    printf("Fibonacci number %d has address %p\n", Arr[I], &Arr[I]);
    Sum += Arr[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number %d has address %p\n", I, &I);
  // CHECK-FIXES-NEXT: Sum += I + 2;
}

const int *constArray() {
  for (int I = 0; I < N; ++I) {
    printf("2 * %d = %d\n", ConstArr[I], ConstArr[I] + ConstArr[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : ConstArr)
  // CHECK-FIXES-NEXT: printf("2 * %d = %d\n", I, I + I);

  const NonTriviallyCopyable NonCopy[N]{};
  for (int I = 0; I < N; ++I) {
    printf("2 * %d = %d\n", NonCopy[I].X, NonCopy[I].X + NonCopy[I].X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & I : NonCopy)
  // CHECK-FIXES-NEXT: printf("2 * %d = %d\n", I.X, I.X + I.X);

  const TriviallyCopyableButBig Big[N]{};
  for (int I = 0; I < N; ++I) {
    printf("2 * %d = %d\n", Big[I].X, Big[I].X + Big[I].X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & I : Big)
  // CHECK-FIXES-NEXT: printf("2 * %d = %d\n", I.X, I.X + I.X);

  bool Something = false;
  for (int I = 0; I < N; ++I) {
    if (Something)
      return &ConstArr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const int & I : ConstArr)
  // CHECK-FIXES-NEXT: if (Something)
  // CHECK-FIXES-NEXT: return &I;
}

struct HasArr {
  int Arr[N];
  Val ValArr[N];
  void implicitThis() {
    for (int I = 0; I < N; ++I) {
      printf("%d", Arr[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int I : Arr)
    // CHECK-FIXES-NEXT: printf("%d", I);

    for (int I = 0; I < N; ++I) {
      printf("%d", ValArr[I].X);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : ValArr)
    // CHECK-FIXES-NEXT: printf("%d", I.X);
  }

  void explicitThis() {
    for (int I = 0; I < N; ++I) {
      printf("%d", this->Arr[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int I : this->Arr)
    // CHECK-FIXES-NEXT: printf("%d", I);

    for (int I = 0; I < N; ++I) {
      printf("%d", this->ValArr[I].X);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : this->ValArr)
    // CHECK-FIXES-NEXT: printf("%d", I.X);
  }
};

struct HasIndirectArr {
  HasArr HA;
  void implicitThis() {
    for (int I = 0; I < N; ++I) {
      printf("%d", HA.Arr[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int I : HA.Arr)
    // CHECK-FIXES-NEXT: printf("%d", I);

    for (int I = 0; I < N; ++I) {
      printf("%d", HA.ValArr[I].X);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : HA.ValArr)
    // CHECK-FIXES-NEXT: printf("%d", I.X);
  }

  void explicitThis() {
    for (int I = 0; I < N; ++I) {
      printf("%d", this->HA.Arr[I]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int I : this->HA.Arr)
    // CHECK-FIXES-NEXT: printf("%d", I);

    for (int I = 0; I < N; ++I) {
      printf("%d", this->HA.ValArr[I].X);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : this->HA.ValArr)
    // CHECK-FIXES-NEXT: printf("%d", I.X);
  }
};

// Loops whose bounds are value-dependent should not be converted.
template <int N>
void dependentExprBound() {
  for (int I = 0; I < N; ++I)
    Arr[I] = 0;
}
template void dependentExprBound<20>();

void memberFunctionPointer() {
  Val V;
  void (Val::*mfpArr[N])(void) = {&Val::g};
  for (int I = 0; I < N; ++I)
    (V.*mfpArr[I])();
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : mfpArr)
  // CHECK-FIXES-NEXT: (V.*I)();

  struct Foo {
    int (Val::*f)();
  } Foo[N];

  for (int I = 0; I < N; ++I)
    int R = (V.*(Foo[I].f))();
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Foo)
  // CHECK-FIXES-NEXT: int R = (V.*(I.f))();

}

} // namespace Array

namespace Iterator {

void f() {
  /// begin()/end() - based for loops here:
  T Tt;
  for (T::iterator It = Tt.begin(), E = Tt.end(); It != E; ++It) {
    printf("I found %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Tt)
  // CHECK-FIXES-NEXT: printf("I found %d\n", It);

  // Do not crash because of Qq.begin() converting. Q::iterator converts with a
  // conversion operator, which has no name, to Q::const_iterator.
  Q Qq;
  for (Q::const_iterator It = Qq.begin(), E = Qq.end(); It != E; ++It) {
    printf("I found %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Qq)
  // CHECK-FIXES-NEXT: printf("I found %d\n", It);

  T *Pt;
  for (T::iterator It = Pt->begin(), E = Pt->end(); It != E; ++It) {
    printf("I found %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : *Pt)
  // CHECK-FIXES-NEXT: printf("I found %d\n", It);

  S Ss;
  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  S *Ps;
  for (S::iterator It = Ps->begin(), E = Ps->end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & P : *Ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", P.X);

  for (S::const_iterator It = Ss.cbegin(), E = Ss.cend(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    printf("s has value %d\n", It->X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    It->X = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: It.X = 3;

  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    (*It).X = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: It.X = 3;

  for (S::iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    It->nonConstFun(4, 5);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: It.nonConstFun(4, 5);

  U Uu;
  for (U::iterator It = Uu.begin(), E = Uu.end(); It != E; ++It) {
    printf("s has value %d\n", It->X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Uu)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  for (U::iterator It = Uu.begin(), E = Uu.end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Uu)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  for (U::iterator It = Uu.begin(), E = Uu.end(); It != E; ++It) {
    Val* a = It.operator->();
  }

  U::iterator A;
  for (U::iterator I = Uu.begin(), E = Uu.end(); I != E; ++I)
    int K = A->X + I->X;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Uu)
  // CHECK-FIXES-NEXT: int K = A->X + I.X;

  dependent<int> V;
  for (dependent<int>::iterator It = V.begin(), E = V.end();
       It != E; ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);

  for (dependent<int>::iterator It(V.begin()), E = V.end();
       It != E; ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);

  doublyDependent<int, int> Intmap;
  for (doublyDependent<int, int>::iterator It = Intmap.begin(), E = Intmap.end();
       It != E; ++It) {
    printf("Intmap[%d] = %d", It->first, It->second);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Intmap)
  // CHECK-FIXES: printf("Intmap[%d] = %d", It.first, It.second);

  // PtrSet's iterator dereferences by value so auto & can't be used.
  {
    PtrSet<int *> Val_int_ptrs;
    for (PtrSet<int *>::iterator I = Val_int_ptrs.begin(),
                                 E = Val_int_ptrs.end();
         I != E; ++I) {
      (void) *I;
    }
    // CHECK-MESSAGES: :[[@LINE-5]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto Val_int_ptr : Val_int_ptrs)
  }

  // This container uses an iterator where the dereference type is a typedef of
  // a reference type. Make sure non-const auto & is still used. A failure here
  // means canonical types aren't being tested.
  {
    TypedefDerefContainer<int> Int_ptrs;
    for (TypedefDerefContainer<int>::iterator I = Int_ptrs.begin(),
                                              E = Int_ptrs.end();
         I != E; ++I) {
      (void) *I;
    }
    // CHECK-MESSAGES: :[[@LINE-5]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (int & Int_ptr : Int_ptrs)
  }

  {
    // Iterators returning an rvalue reference should disqualify the loop from
    // transformation.
    RValueDerefContainer<int> Container;
    for (RValueDerefContainer<int>::iterator I = Container.begin(),
                                             E = Container.end();
         I != E; ++I) {
      (void) *I;
    }
  }

  dependent<Val *> Dpp;
  for (dependent<Val *>::iterator I = Dpp.begin(), E = Dpp.end(); I != E; ++I) {
    printf("%d\n", (**I).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Dpp)
  // CHECK-FIXES-NEXT: printf("%d\n", (*I).X);

  for (dependent<Val *>::iterator I = Dpp.begin(), E = Dpp.end(); I != E; ++I) {
    printf("%d\n", (*I)->X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & I : Dpp)
  // CHECK-FIXES-NEXT: printf("%d\n", I->X);

  for (S::iterator It = begin(Ss), E = end(Ss); It != E; ++It) {
    printf("s0 has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: printf("s0 has value %d\n", It.X);

  for (S::iterator It = std::begin(Ss), E = std::end(Ss); It != E; ++It) {
    printf("s1 has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : Ss)
  // CHECK-FIXES-NEXT: printf("s1 has value %d\n", It.X);

  for (S::iterator It = begin(*Ps), E = end(*Ps); It != E; ++It) {
    printf("s2 has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : *Ps)
  // CHECK-FIXES-NEXT: printf("s2 has value %d\n", It.X);

  for (S::iterator It = begin(*Ps); It != end(*Ps); ++It) {
    printf("s3 has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & It : *Ps)
  // CHECK-FIXES-NEXT: printf("s3 has value %d\n", It.X);

  for (S::const_iterator It = cbegin(Ss), E = cend(Ss); It != E; ++It) {
    printf("s4 has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto It : Ss)
  // CHECK-FIXES-NEXT: printf("s4 has value %d\n", It.X);
}

// Tests to verify the proper use of auto where the init variable type and the
// initializer type differ or are mostly the same except for const qualifiers.
void different_type() {
  // Ss.begin() returns a type 'iterator' which is just a non-const pointer and
  // differs from const_iterator only on the const qualification.
  S Ss;
  for (S::const_iterator It = Ss.begin(), E = Ss.end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto It : Ss)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", It.X);

  S *Ps;
  for (S::const_iterator It = Ps->begin(), E = Ps->end(); It != E; ++It) {
    printf("s has value %d\n", (*It).X);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto P : *Ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", P.X);

  dependent<int> V;
  for (dependent<int>::const_iterator It = V.begin(), E = V.end();
       It != E; ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);

  for (dependent<int>::const_iterator It(V.begin()), E = V.end();
       It != E; ++It) {
    printf("Fibonacci number is %d\n", *It);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int It : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", It);
}

// Tests to ensure that an implicit 'this' is picked up as the container.
// If member calls are made to 'this' within the loop, the transform becomes
// risky as these calls may affect state that affects the loop.
class C {
public:
  typedef MutableVal *iterator;
  typedef const MutableVal *const_iterator;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  void doSomething();
  void doSomething() const;

  void doLoop() {
    for (iterator I = begin(), E = end(); I != E; ++I)
      (void) *I;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : *this)

    for (iterator I = C::begin(), E = C::end(); I != E; ++I)
      (void) *I;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : *this)

    for (iterator I = begin(), E = end(); I != E; ++I) {
      (void) *I;
      doSomething();
    }

    for (iterator I = begin(); I != end(); ++I)
      (void) *I;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : *this)

    for (iterator I = begin(); I != end(); ++I) {
      (void) *I;
      doSomething();
    }
  }

  void doLoop() const {
    for (const_iterator I = begin(), E = end(); I != E; ++I)
      (void) *I;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto I : *this)

    for (const_iterator I = C::begin(), E = C::end(); I != E; ++I)
      (void) *I;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto I : *this)

    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      (void) *I;
      doSomething();
    }
  }
};

class C2 {
public:
  typedef MutableVal *iterator;

  iterator begin() const;
  iterator end() const;

  void doLoop() {
    // The implicit 'this' will have an Implicit cast to const C2* wrapped
    // around it. Make sure the replacement still happens.
    for (iterator I = begin(), E = end(); I != E; ++I)
      (void) *I;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & I : *this)
  }
};

} // namespace Iterator

namespace PseudoArray {

const int N = 6;
dependent<int> V;
dependent<int> *Pv;
const dependent<NonTriviallyCopyable> Constv;
const dependent<NonTriviallyCopyable> *Pconstv;

transparent<dependent<int>> Cv;
dependent_derived<int> VD;

void f() {
  int Sum = 0;
  for (int I = 0, E = V.size(); I < E; ++I) {
    printf("Fibonacci number is %d\n", V[I]);
    Sum += V[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = V.size(); I < E; ++I) {
    printf("Fibonacci number is %d\n", V.at(I));
    Sum += V.at(I) + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = Pv->size(); I < E; ++I) {
    printf("Fibonacci number is %d\n", Pv->at(I));
    Sum += Pv->at(I) + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : *Pv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  // This test will fail if size() isn't called repeatedly, since it
  // returns unsigned int, and 0 is deduced to be signed int.
  // FIXME: Insert the necessary explicit conversion, or write out the types
  // explicitly.
  for (int I = 0; I < Pv->size(); ++I) {
    printf("Fibonacci number is %d\n", (*Pv).at(I));
    Sum += (*Pv)[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : *Pv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0; I < Cv->size(); ++I) {
    printf("Fibonacci number is %d\n", Cv->at(I));
    Sum += Cv->at(I) + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : *Cv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = V.size(); E > I; ++I) {
    printf("Fibonacci number is %d\n", V[I]);
    Sum += V[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = V.size(); I != E; ++I) {
    printf("Fibonacci number is %d\n", V[I]);
    Sum += V[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = V.size(); E != I; ++I) {
    printf("Fibonacci number is %d\n", V[I]);
    Sum += V[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = VD.size(); E != I; ++I) {
    printf("Fibonacci number is %d\n", VD[I]);
    Sum += VD[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : VD)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = size(V); E != I; ++I) {
    printf("Fibonacci number is %d\n", V[I]);
    Sum += V[I] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : V)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I);
  // CHECK-FIXES-NEXT: Sum += I + 2;

  for (int I = 0, E = size(V); E != I; ++I) {
    V[I] = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : V)
  // CHECK-FIXES-NEXT: I = 0;

  for (int I = 0, E = std::size(V); E != I; ++I) {
    V[I] = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : V)
  // CHECK-FIXES-NEXT: I = 0;

  // Although 'length' might be a valid free function, only size() is standardized
  for (int I = 0, E = length(V); E != I; ++I) {
    printf("Fibonacci number is %d\n", V[I]);
    Sum += V[I] + 2;
  }

  dependent<Val> Vals;
  for (int I = 0, E = size(Vals); E != I; ++I) {
    Sum += Vals[I].X;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Val : Vals)
  // CHECK-FIXES-NEXT: Sum += Val.X;
}

// Ensure that 'const auto &' is used with containers of non-trivial types.
void constness() {
  int Sum = 0;
  for (int I = 0, E = Constv.size(); I < E; ++I) {
    printf("Fibonacci number is %d\n", Constv[I].X);
    Sum += Constv[I].X + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & I : Constv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I.X);
  // CHECK-FIXES-NEXT: Sum += I.X + 2;

  for (int I = 0, E = Constv.size(); I < E; ++I) {
    printf("Fibonacci number is %d\n", Constv.at(I).X);
    Sum += Constv.at(I).X + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & I : Constv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I.X);
  // CHECK-FIXES-NEXT: Sum += I.X + 2;

  for (int I = 0, E = Pconstv->size(); I < E; ++I) {
    printf("Fibonacci number is %d\n", Pconstv->at(I).X);
    Sum += Pconstv->at(I).X + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & I : *Pconstv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I.X);
  // CHECK-FIXES-NEXT: Sum += I.X + 2;

  // This test will fail if size() isn't called repeatedly, since it
  // returns unsigned int, and 0 is deduced to be signed int.
  // FIXME: Insert the necessary explicit conversion, or write out the types
  // explicitly.
  for (int I = 0; I < Pconstv->size(); ++I) {
    printf("Fibonacci number is %d\n", (*Pconstv).at(I).X);
    Sum += (*Pconstv)[I].X + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & I : *Pconstv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", I.X);
  // CHECK-FIXES-NEXT: Sum += I.X + 2;
}

void constRef(const dependent<int>& ConstVRef) {
  int sum = 0;
  // FIXME: This does not work with size_t (probably due to the implementation
  // of dependent); make dependent work exactly like a std container type.
  for (int I = 0; I < ConstVRef.size(); ++I) {
    sum += ConstVRef[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : ConstVRef)
  // CHECK-FIXES-NEXT: sum += I;

  for (auto I = ConstVRef.begin(), E = ConstVRef.end(); I != E; ++I) {
    sum += *I;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : ConstVRef)
  // CHECK-FIXES-NEXT: sum += I;
}

// Check for loops that don't mention containers.
void noContainer() {
  for (auto I = 0; I < V.size(); ++I) {
  }

  for (auto I = 0; I < V.size(); ++I)
    ;
}

struct NoBeginEnd {
  unsigned size() const;
  unsigned& operator[](int);
  const unsigned& operator[](int) const;
};

struct NoConstBeginEnd {
  NoConstBeginEnd();
  unsigned size() const;
  unsigned* begin();
  unsigned* end();
  unsigned& operator[](int);
  const unsigned& operator[](int) const;
};

struct ConstBeginEnd {
  ConstBeginEnd();
  unsigned size() const;
  unsigned* begin() const;
  unsigned* end() const;
  unsigned& operator[](int);
  const unsigned& operator[](int) const;
};

// Shouldn't transform pseudo-array uses if the container doesn't provide
// begin() and end() of the right const-ness.
void NoBeginEndTest() {
  NoBeginEnd NBE;
  for (unsigned I = 0, E = NBE.size(); I < E; ++I)
    printf("%d\n", NBE[I]);

  const NoConstBeginEnd Const_NCBE;
  for (unsigned I = 0, E = Const_NCBE.size(); I < E; ++I)
    printf("%d\n", Const_NCBE[I]);

  ConstBeginEnd CBE;
  for (unsigned I = 0, E = CBE.size(); I < E; ++I)
    printf("%d\n", CBE[I]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (unsigned int I : CBE)
  // CHECK-FIXES-NEXT: printf("%d\n", I);

  const ConstBeginEnd Const_CBE;
  for (unsigned I = 0, E = Const_CBE.size(); I < E; ++I)
    printf("%d\n", Const_CBE[I]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (unsigned int I : Const_CBE)
  // CHECK-FIXES-NEXT: printf("%d\n", I);
}

struct DerefByValue {
  DerefByValue();
  struct iter { unsigned operator*(); };
  unsigned size() const;
  iter begin();
  iter end();
  unsigned operator[](int);
};

void derefByValueTest() {
  DerefByValue DBV;
  for (unsigned I = 0, E = DBV.size(); I < E; ++I) {
    printf("%d\n", DBV[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (unsigned int I : DBV)
  // CHECK-FIXES-NEXT: printf("%d\n", I);

  for (unsigned I = 0, E = DBV.size(); I < E; ++I) {
    auto f = [DBV, I]() {};
    printf("%d\n", DBV[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (unsigned int I : DBV)
  // CHECK-FIXES-NEXT: auto f = [DBV, &I]() {};
  // CHECK-FIXES-NEXT: printf("%d\n", I);
}

void fundamentalTypesTest() {
  const int N = 10;
  bool Bools[N];
  for (int i = 0; i < N; ++i)
    printf("%d", Bools[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (bool Bool : Bools)

  int Ints[N];
  unsigned short int Shorts[N];
  for (int i = 0; i < N; ++i)
    printf("%d", Shorts[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (unsigned short Short : Shorts)

  signed long Longs[N];
  for (int i = 0; i < N; ++i)
    printf("%d", Longs[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (long Long : Longs)

  long long int LongLongs[N];
  for (int i = 0; i < N; ++i)
    printf("%d", LongLongs[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (long long LongLong : LongLongs)

  char Chars[N];
  for (int i = 0; i < N; ++i)
    printf("%d", Chars[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (char Char : Chars)

  wchar_t WChars[N];
  for (int i = 0; i < N; ++i)
    printf("%d", WChars[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (wchar_t WChar : WChars)

  float Floats[N];
  for (int i = 0; i < N; ++i)
    printf("%d", Floats[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (float Float : Floats)

  double Doubles[N];
  for (int i = 0; i < N; ++i)
    printf("%d", Doubles[i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (double Double : Doubles)
}

template <unsigned  p> void _dependenceArrayTest() {
  unsigned test[3][p];
  for (unsigned i = 0; i < p; ++i)
    for (unsigned j = 0; j < 3; ++j)
      printf("%d", test[j][i]);
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
  // CHECK-FIXES: (auto & j : test)
  // CHECK-FIXES: printf("%d", j[i]);
}
void dependenceArrayTest() {
  _dependenceArrayTest<1>();
  _dependenceArrayTest<2>();
}

} // namespace PseudoArray

namespace PR78381 {
  struct blocked_range {
    int begin() const;
    int end() const;
  };

  void test() {
    blocked_range r;
    for (auto i = r.begin(); i!=r.end(); ++i) {
    }
  }
}
