// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cstddef>

template <typename T> void initialize(T *Ptr, size_t Count, T Value) {
  for (size_t I = 0; I < Count; ++I)
    Ptr[I] = Value;
}

int main() {
  sycl::queue Q;

  constexpr int Dims = 3;
  const sycl::range<Dims> LocalRange{2, 3, 1};
  const sycl::range<Dims> GroupRange{1, 2, 3};
  const sycl::range<Dims> GlobalRange = LocalRange * GroupRange;
  const size_t DataLen = GlobalRange.size();

  size_t *GroupRangeData = sycl::malloc_shared<size_t>(DataLen * Dims, Q);
  size_t *GroupLinearIdData = sycl::malloc_shared<size_t>(DataLen * Dims, Q);
  initialize(GroupRangeData, DataLen * Dims, static_cast<size_t>(0));
  initialize(GroupLinearIdData, DataLen * Dims, static_cast<size_t>(0));

  Q.parallel_for<class group_get_group_range_regression>(
      sycl::nd_range<3>{GlobalRange, LocalRange}, [=](sycl::nd_item<3> It) {
        const size_t Off = It.get_global_linear_id() * Dims;
        const auto GR = It.get_group().get_group_range();
        GroupRangeData[Off + 0] = GR[0];
        GroupRangeData[Off + 1] = GR[1];
        GroupRangeData[Off + 2] = GR[2];
      });

  Q.parallel_for<class group_get_group_linear_id_regression>(
      sycl::nd_range<3>{GlobalRange, LocalRange}, [=](sycl::nd_item<3> It) {
        const size_t Off = It.get_global_linear_id() * Dims;
        const size_t LI = It.get_group().get_group_linear_id();
        GroupLinearIdData[Off + 0] = LI;
        GroupLinearIdData[Off + 1] = LI;
        GroupLinearIdData[Off + 2] = LI;
      });

  Q.wait();

  const size_t SizeZ = GlobalRange.get(0);
  const size_t SizeY = GlobalRange.get(1);
  const size_t SizeX = GlobalRange.get(2);

  bool Fail = false;
  for (size_t Z = 0; Z < SizeZ; ++Z) {
    for (size_t Y = 0; Y < SizeY; ++Y) {
      for (size_t X = 0; X < SizeX; ++X) {
        const size_t Ind = Z * SizeX * SizeY + Y * SizeX + X;

        const size_t Off = Ind * Dims;
        Fail |= GroupRangeData[Off + 0] != GroupRange.get(0);
        Fail |= GroupRangeData[Off + 1] != GroupRange.get(1);
        Fail |= GroupRangeData[Off + 2] != GroupRange.get(2);

        const sycl::id<3> GlobalId{Z, Y, X};
        const sycl::id<3> GroupId = GlobalId / LocalRange;
        const size_t GoldLinearId =
            GroupId.get(0) * GroupRange.get(1) * GroupRange.get(2) +
            GroupId.get(1) * GroupRange.get(2) + GroupId.get(2);
        Fail |= GroupLinearIdData[Off + 0] != GoldLinearId;
        Fail |= GroupLinearIdData[Off + 1] != GoldLinearId;
        Fail |= GroupLinearIdData[Off + 2] != GoldLinearId;
      }
    }
  }

  sycl::free(GroupRangeData, Q);
  sycl::free(GroupLinearIdData, Q);
  return Fail;
}
