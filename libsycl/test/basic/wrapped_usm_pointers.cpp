// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <iostream>

struct Simple {
  int *Data;
  int Addition;
};

struct WrapperOfSimple {
  int Addition;
  Simple Obj;
};

struct NonTrivial {
  int Addition;
  int *Data;

  NonTrivial(int *D, int A) : Addition(A), Data(D) {}
};

struct NonTrivialDerived : NonTrivial {
  int AA = 0;
  NonTrivialDerived(int *D, int A) : NonTrivial(D, A) {}
};

using namespace sycl;

int main() {
  constexpr int NumOfElements = 7;

  queue Q;

  NonTrivial NonTrivialObj(sycl::malloc_shared<int>(NumOfElements, Q), 38);
  NonTrivialDerived NonTrivialDerivedObj(
      sycl::malloc_shared<int>(NumOfElements, Q), 39);
  Simple SimpleObj = {sycl::malloc_shared<int>(NumOfElements, Q), 42};
  WrapperOfSimple WrapperOfSimpleObj = {
      300, {sycl::malloc_shared<int>(NumOfElements, Q), 100500}};

  // Test simple struct containing pointer.
  Q.parallel_for(NumOfElements, [=](id<1> Idx) {
    SimpleObj.Data[Idx] = Idx + SimpleObj.Addition;
  });

  // Test simple non-trivial struct containing pointer.
  Q.parallel_for(NumOfElements, [=](id<1> Idx) {
    NonTrivialObj.Data[Idx] = Idx + NonTrivialObj.Addition;
  });

  // Test simple non-trivial derived struct containing pointer.
  Q.parallel_for(NumOfElements, [=](id<1> Idx) {
    NonTrivialDerivedObj.Data[Idx] = Idx + NonTrivialDerivedObj.Addition;
  });

  // Test nested struct containing pointer.
  Q.parallel_for(NumOfElements, [=](id<1> Idx) {
    WrapperOfSimpleObj.Obj.Data[Idx] = Idx + WrapperOfSimpleObj.Obj.Addition;
  });

  // Test array of structs containing pointers.
  Simple SimpleArr[NumOfElements];
  for (int I = 0; I < NumOfElements; ++I) {
    SimpleArr[I].Data = sycl::malloc_shared<int>(NumOfElements, Q);
    SimpleArr[I].Addition = 38 + I;
  }

  Q.parallel_for(range<2>(NumOfElements, NumOfElements), [=](item<2> Idx) {
    SimpleArr[Idx.get_id(0)].Data[Idx.get_id(1)] =
        Idx.get_id(1) + SimpleArr[Idx.get_id(0)].Addition;
  });

  Q.wait();

  auto Checker = [](auto Obj) {
    for (int I = 0; I < NumOfElements; ++I) {
      if (Obj.Data[I] != (I + Obj.Addition)) {
        std::cout << "line: " << __LINE__ << " result[" << I << "] is "
                  << Obj.Data[I] << " expected " << I + Obj.Addition
                  << std::endl;
        return true; // true if fail
      }
    }

    return false;
  };

  bool Fail = false;
  Fail |= Checker(SimpleObj);
  Fail |= Checker(NonTrivialObj);
  Fail |= Checker(NonTrivialDerivedObj);
  Fail |= Checker(WrapperOfSimpleObj.Obj);

  for (int I = 0; I < NumOfElements; ++I)
    Fail |= Checker(SimpleArr[I]);

  // Free allocated memory.
  sycl::free(NonTrivialObj.Data, Q);
  sycl::free(NonTrivialDerivedObj.Data, Q);
  sycl::free(SimpleObj.Data, Q);
  sycl::free(WrapperOfSimpleObj.Obj.Data, Q);

  for (int I = 0; I < NumOfElements; ++I)
    sycl::free(SimpleArr[I].Data, Q);

  return Fail;
}
