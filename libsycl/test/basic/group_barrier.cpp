// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>

constexpr int LocalSize = 8;
constexpr int WorkGroups = 4;
constexpr int GlobalSize = WorkGroups * LocalSize;

static bool runBarrierCase(sycl::queue &Q, int GroupBias, bool MaxCase) {
  int *Data = sycl::malloc_shared<int>(GlobalSize, Q);
  int *LocalData = sycl::malloc_shared<int>(GlobalSize, Q);

  Q.parallel_for<class barrier_kernel>(
      sycl::nd_range<1>{GlobalSize, LocalSize}, [=](sycl::nd_item<1> It) {
        const int Lid = It.get_local_id(0);
        const int Gid = It.get_group().get_group_id(0);
        int *GroupData = LocalData + Gid * LocalSize;
        GroupData[Lid] = Gid * GroupBias + Lid + 1;
        sycl::group_barrier(It.get_group());

        if (Lid == 0) {
          int Result = GroupData[0];
          if (MaxCase) {
            for (int I = 1; I < LocalSize; ++I)
              Result = std::max(Result, GroupData[I]);
          } else {
            for (int I = 1; I < LocalSize; ++I)
              Result += GroupData[I];
          }
          GroupData[0] = Result;
        }

        sycl::group_barrier(It.get_group());
        Data[It.get_global_id(0)] = GroupData[0];
      });

  Q.wait();

  bool Failure = false;
  for (int Gid = 0; Gid < WorkGroups; ++Gid) {
    int Expected = 0;
    for (int Lid = 0; Lid < LocalSize; ++Lid) {
      const int Value = Gid * GroupBias + Lid + 1;
      Expected += Value;
    }
    if (MaxCase)
      Expected = Gid * GroupBias + LocalSize;

    for (int Lid = 0; Lid < LocalSize; ++Lid) {
      const int Index = Gid * LocalSize + Lid;
      if (Data[Index] != Expected) {
        std::cerr << "Mismatch at group " << Gid << " lane " << Lid << ": got "
                  << Data[Index] << ", expected " << Expected << std::endl;
        Failure = true;
      }
    }
  }

  sycl::free(Data, Q);
  sycl::free(LocalData, Q);
  return !Failure;
}

int main() {
  sycl::queue Q;

  if (!runBarrierCase(Q, 10, false))
    return 1;
  if (!runBarrierCase(Q, 17, true))
    return 1;

  return 0;
}
