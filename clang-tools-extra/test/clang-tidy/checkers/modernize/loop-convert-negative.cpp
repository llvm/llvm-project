// RUN: %check_clang_tidy %s modernize-loop-convert %t -- -- -I %S/Inputs/loop-convert

#include "structures.h"

// CHECK-FIXES-NOT: for ({{.*[^:]:[^:].*}})
// CHECK-MESSAGES-NOT: modernize-loop-convert

namespace somenamespace {
  template <class T> auto begin(T& t) -> decltype(t.begin());
  template <class T> auto begin(const T& t) -> decltype(t.begin());
  template <class T> auto end(T& t) -> decltype(t.end());
  template <class T> auto end(const T& t) -> decltype(t.end());
  template <class T> auto size(const T& t) -> decltype(t.size());
} // namespace somenamespace

struct SomeClass {
  template <class T> static auto begin(T& t) -> decltype(t.begin());
  template <class T> static auto begin(const T& t) -> decltype(t.begin());
  template <class T> static auto end(T& t) -> decltype(t.end());
  template <class T> static auto end(const T& t) -> decltype(t.end());
  template <class T> static auto size(const T& t) -> decltype(t.size());
};

namespace Negative {

const int N = 6;
int Arr[N] = {1, 2, 3, 4, 5, 6};
int (*pArr)[N] = &Arr;
int Sum = 0;

// Checks for the Index start and end:
void IndexStartAndEnd() {
  for (int I = 0; I < N + 1; ++I)
    Sum += Arr[I];

  for (int I = 0; I < N - 1; ++I)
    Sum += Arr[I];

  for (int I = 1; I < N; ++I)
    Sum += Arr[I];

  for (int I = 1; I < N; ++I)
    Sum += Arr[I];

  for (int I = 0;; ++I)
    Sum += (*pArr)[I];
}

// Checks for invalid increment steps:
void increment() {
  for (int I = 0; I < N; --I)
    Sum += Arr[I];

  for (int I = 0; I < N; I)
    Sum += Arr[I];

  for (int I = 0; I < N;)
    Sum += Arr[I];

  for (int I = 0; I < N; I += 2)
    Sum++;
}

// Checks to make sure that the Index isn't used outside of the array:
void IndexUse() {
  for (int I = 0; I < N; ++I)
    Arr[I] += 1 + I;
}

// Check for loops that don't mention arrays
void noArray() {
  for (int I = 0; I < N; ++I)
    Sum += I;

  for (int I = 0; I < N; ++I) {
  }

  for (int I = 0; I < N; ++I)
    ;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int BadIndex;
  for (int I = 0; BadIndex < N; ++I)
    Sum += Arr[I];

  for (int I = 0; I < N; ++BadIndex)
    Sum += Arr[I];

  for (int I = 0; BadIndex < N; ++BadIndex)
    Sum += Arr[I];

  for (int I = 0; BadIndex < N; ++BadIndex)
    Sum += Arr[BadIndex];
}

// Checks for multiple arrays Indexed.
void multipleArrays() {
  int BadArr[N];

  for (int I = 0; I < N; ++I)
    Sum += Arr[I] + BadArr[I];

  for (int I = 0; I < N; ++I) {
    int K = BadArr[I];
    Sum += Arr[I] + K;
  }
}

} // namespace Negative

namespace NegativeIterator {

S Ss;
T Tt;
U Tu;

struct BadBeginEnd : T {
  iterator notBegin();
  iterator notEnd();
  iterator begin(int);
  iterator end(int);
  iterator begin();
  iterator end();
};

void notBeginOrEnd() {
  BadBeginEnd Bad;
  for (T::iterator I = Bad.notBegin(), E = Bad.end();  I != E; ++I)
    int K = *I;

  for (T::iterator I = Bad.begin(), E = Bad.notEnd();  I != E; ++I)
    int K = *I;

  for (T::iterator I = Bad.begin(0), E = Bad.end(0);  I != E; ++I)
    int K = *I;
}

void badLoopShapes() {
  for (T::iterator I = Tt.begin(), E = Tt.end(), F = E;  I != E; ++I)
    int K = *I;

  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E;)
    int K = *I;

  for (T::iterator I = Tt.begin(), E = Tt.end();; ++I)
    int K = *I;

  T::iterator OutsideI;
  T::iterator OutsideE;

  for (; OutsideI != OutsideE; ++OutsideI)
    int K = *OutsideI;
}

void iteratorArrayMix() {
  int Lower;
  const int N = 6;
  for (T::iterator I = Tt.begin(), E = Tt.end(); Lower < N; ++I)
    int K = *I;

  for (T::iterator I = Tt.begin(), E = Tt.end(); Lower < N; ++Lower)
    int K = *I;
}

struct ExtraConstructor : T::iterator {
  ExtraConstructor(T::iterator, int);
  explicit ExtraConstructor(T::iterator);
};

void badConstructor() {
  for (T::iterator I = ExtraConstructor(Tt.begin(), 0), E = Tt.end();
        I != E; ++I)
    int K = *I;
  for (T::iterator I = ExtraConstructor(Tt.begin()), E = Tt.end();  I != E; ++I)
    int K = *I;
}

void foo(S::iterator It) {}
class Foo {public: void bar(S::iterator It); };
Foo Fo;

void iteratorUsed() {
  for (S::iterator I = Ss.begin(), E = Ss.end();  I != E; ++I)
    foo(I);

  for (S::iterator I = Ss.begin(), E = Ss.end();  I != E; ++I)
    Fo.bar(I);

  S::iterator Ret;
  for (S::iterator I = Ss.begin(), E = Ss.end();  I != E; ++I)
    Ret = I;
}

void iteratorMemberUsed() {
  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    I.X = *I;

  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    int K = I.X + *I;

  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    int K = E.X + *I;
}

void iteratorMethodCalled() {
  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    I.insert(3);

  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    if (I != I)
      int K = 3;
}

void iteratorOperatorCalled() {
  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    int K = *(++I);

  for (S::iterator I = Ss.begin(), E = Ss.end();  I != E; ++I)
    MutableVal K = *(++I);
}

void differentContainers() {
  T Other;
  for (T::iterator I = Tt.begin(), E = Other.end();  I != E; ++I)
    int K = *I;
  for (T::iterator I = begin(Tt), E = end(Other);  I != E; ++I)
    int K = *I;

  for (T::iterator I = Other.begin(), E = Tt.end();  I != E; ++I)
    int K = *I;

  S OtherS;
  for (S::iterator I = Ss.begin(), E = OtherS.end();  I != E; ++I)
    MutableVal K = *I;

  for (S::iterator I = OtherS.begin(), E = Ss.end();  I != E; ++I)
    MutableVal K = *I;
}

void mixedMemberAndADL() {
  for (T::iterator I = Tt.begin(), E = end(Tt);  I != E; ++I)
    int K = *I;
  for (T::iterator I = begin(Tt), E = Tt.end();  I != E; ++I)
    int K = *I;
  for (T::iterator I = std::begin(Tt), E = Tt.end();  I != E; ++I)
    int K = *I;
  for (T::iterator I = std::begin(Tt), E = end(Tt);  I != E; ++I)
    int K = *I;
}

void nonADLOrStdCalls() {
  for (T::iterator I = SomeClass::begin(Tt), E = SomeClass::end(Tt);  I != E; ++I)
    int K = *I;
  for (T::iterator I = somenamespace::begin(Tt), E = somenamespace::end(Tt);  I != E; ++I)
    int K = *I;
}

void wrongIterators() {
  T::iterator Other;
  for (T::iterator I = Tt.begin(), E = Tt.end(); I != Other; ++I)
    int K = *I;
}

struct EvilArrow : U {
  // Please, no one ever write code like this.
  U *operator->();
};

void differentMemberAccessTypes() {
  EvilArrow A;
  for (EvilArrow::iterator I = A.begin(), E = A->end();  I != E; ++I)
    Val K = *I;
  for (EvilArrow::iterator I = A->begin(), E = A.end();  I != E; ++I)
    Val K = *I;
}

void f(const T::iterator &It, int);
void f(const T &It, int);
void g(T &It, int);

void iteratorPassedToFunction() {
  for (T::iterator I = Tt.begin(), E = Tt.end();  I != E; ++I)
    f(I, *I);
}

// FIXME: These tests can be removed if this tool ever does enough analysis to
// decide that this is a safe transformation. Until then, we don't want it
// applied.
void iteratorDefinedOutside() {
  T::iterator TheEnd = Tt.end();
  for (T::iterator I = Tt.begin(); I != TheEnd; ++I)
    int K = *I;

  T::iterator TheBegin = Tt.begin();
  for (T::iterator E = Tt.end(); TheBegin != E; ++TheBegin)
    int K = *TheBegin;
}

} // namespace NegativeIterator

namespace NegativePseudoArray {

const int N = 6;
dependent<int> V;
dependent<int> *Pv;

int Sum = 0;

// Checks for the Index start and end:
void IndexStartAndEnd() {
  for (int I = 0; I < V.size() + 1; ++I)
    Sum += V[I];

  for (int I = 0; I < V.size() - 1; ++I)
    Sum += V[I];

  for (int I = 1; I < V.size(); ++I)
    Sum += V[I];

  for (int I = 1; I < V.size(); ++I)
    Sum += V[I];

  for (int I = 0;; ++I)
    Sum += (*Pv)[I];
}

// Checks for invalid increment steps:
void increment() {
  for (int I = 0; I < V.size(); --I)
    Sum += V[I];

  for (int I = 0; I < V.size(); I)
    Sum += V[I];

  for (int I = 0; I < V.size();)
    Sum += V[I];

  for (int I = 0; I < V.size(); I += 2)
    Sum++;
}

// Checks to make sure that the Index isn't used outside of the container:
void IndexUse() {
  for (int I = 0; I < V.size(); ++I)
    V[I] += 1 + I;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int BadIndex;
  for (int I = 0; BadIndex < V.size(); ++I)
    Sum += V[I];

  for (int I = 0; I < V.size(); ++BadIndex)
    Sum += V[I];

  for (int I = 0; BadIndex < V.size(); ++BadIndex)
    Sum += V[I];

  for (int I = 0; BadIndex < V.size(); ++BadIndex)
    Sum += V[BadIndex];
}

// Checks for an array Indexed in addition to the container.
void multipleArrays() {
  int BadArr[N];

  for (int I = 0; I < V.size(); ++I)
    Sum += V[I] + BadArr[I];

  for (int I = 0; I < V.size(); ++I)
    Sum += BadArr[I];

  for (int I = 0; I < V.size(); ++I) {
    int K = BadArr[I];
    Sum += K + 2;
  }

  for (int I = 0; I < V.size(); ++I) {
    int K = BadArr[I];
    Sum += V[I] + K;
  }
}

// Checks for multiple containers being Indexed container.
void multipleContainers() {
  dependent<int> BadArr;

  for (int I = 0; I < V.size(); ++I)
    Sum += V[I] + BadArr[I];

  for (int I = 0; I < V.size(); ++I)
    Sum += BadArr[I];

  for (int I = 0; I < V.size(); ++I) {
    int K = BadArr[I];
    Sum += K + 2;
  }

  for (int I = 0; I < V.size(); ++I) {
    int K = BadArr[I];
    Sum += V[I] + K;
  }
}

// Check to make sure that dereferenced pointers-to-containers behave nicely.
void derefContainer() {
  // Note the dependent<T>::operator*() returns another dependent<T>.
  // This test makes sure that we don't allow an arbitrary number of *'s.
  for (int I = 0; I < Pv->size(); ++I)
    Sum += (**Pv).at(I);

  for (int I = 0; I < Pv->size(); ++I)
    Sum += (**Pv)[I];
}

void wrongEnd() {
  int Bad;
  for (int I = 0, E = V.size(); I < Bad; ++I)
    Sum += V[I];
}

void nonADLOrStdCalls() {
  for (int I = 0, E = somenamespace::size(V); E != I; ++I)
    printf("Fibonacci number is %d\n", V[I]);
  for (int I = 0, E = SomeClass::size(V); E != I; ++I)
    printf("Fibonacci number is %d\n", V[I]);
}

// Checks to see that non-const member functions are not called on the container
// object.
// These could be conceivably allowed with a lower required confidence level.
void memberFunctionCalled() {
  for (int I = 0; I < V.size(); ++I) {
    Sum += V[I];
    V.foo();
  }

  for (int I = 0; I < V.size(); ++I) {
    Sum += V[I];
    dependent<int>::iterator It = V.begin();
  }
}

} // namespace NegativePseudoArray

namespace NegativeMultiEndCall {

S Ss;
T Tt;
U Uu;

void f(X);
void f(S);
void f(T);

void complexContainer() {
  X Xx;
  for (S::iterator I = Xx.Ss.begin(), E = Xx.Ss.end();  I != E; ++I) {
    f(Xx);
    MutableVal K = *I;
  }

  for (T::iterator I = Xx.Tt.begin(), E = Xx.Tt.end();  I != E; ++I) {
    f(Xx);
    int K = *I;
  }

  for (S::iterator I = Xx.Ss.begin(), E = Xx.Ss.end();  I != E; ++I) {
    f(Xx.Ss);
    MutableVal K = *I;
  }

  for (T::iterator I = Xx.Tt.begin(), E = Xx.Tt.end();  I != E; ++I) {
    f(Xx.Tt);
    int K = *I;
  }

  for (S::iterator I = Xx.getS().begin(), E = Xx.getS().end();  I != E; ++I) {
    f(Xx.getS());
    MutableVal K = *I;
  }

  X Exes[5];
  int Index = 0;

  for (S::iterator I = Exes[Index].getS().begin(),
                   E = Exes[Index].getS().end();
        I != E; ++I) {
    Index++;
    MutableVal K = *I;
  }
}

} // namespace NegativeMultiEndCall

namespace NoUsages {

const int N = 6;
int Arr[N] = {1, 2, 3, 4, 5, 6};
S Ss;
dependent<int> V;
int Count = 0;

void foo();

void f() {
  for (int I = 0; I < N; ++I) {}
  for (int I = 0; I < N; ++I)
    printf("Hello world\n");
  for (int I = 0; I < N; ++I)
    ++Count;
  for (int I = 0; I < N; ++I)
    foo();

  for (S::iterator I = Ss.begin(), E = Ss.end(); I != E; ++I) {}
  for (S::iterator I = Ss.begin(), E = Ss.end(); I != E; ++I)
    printf("Hello world\n");
  for (S::iterator I = Ss.begin(), E = Ss.end(); I != E; ++I)
    ++Count;
  for (S::iterator I = Ss.begin(), E = Ss.end(); I != E; ++I)
    foo();

  for (int I = 0; I < V.size(); ++I) {}
  for (int I = 0; I < V.size(); ++I)
    printf("Hello world\n");
  for (int I = 0; I < V.size(); ++I)
    ++Count;
  for (int I = 0; I < V.size(); ++I)
    foo();
}

} // namespace NoUsages
