// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

struct Data {
  unsigned int LocalId;
  unsigned int LocalRange;
  unsigned int MaxLocalRange;
  unsigned int GroupId;
  unsigned int GroupRange;
};

bool check(sycl::queue &Q, unsigned int G, unsigned int L) {
  Data *SyclData = sycl::malloc_shared<Data>(G, Q);
  size_t *SgSize = sycl::malloc_shared<size_t>(1, Q);

  for (unsigned int I = 0; I < G; ++I)
    SyclData[I] = {0, 0, 0, 0, 0};
  SgSize[0] = 0;

  Q.parallel_for<class sycl_subgr_common>(
      sycl::nd_range<1>(sycl::range<1>(G), sycl::range<1>(L)),
      [=](sycl::nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        const unsigned int Index =
            static_cast<unsigned int>(NdItem.get_global_id(0));
        SyclData[Index].LocalId =
            static_cast<unsigned int>(SG.get_local_id()[0]);
        SyclData[Index].LocalRange =
            static_cast<unsigned int>(SG.get_local_range()[0]);
        SyclData[Index].MaxLocalRange =
            static_cast<unsigned int>(SG.get_max_local_range()[0]);
        SyclData[Index].GroupId =
            static_cast<unsigned int>(SG.get_group_id()[0]);
        SyclData[Index].GroupRange =
            static_cast<unsigned int>(SG.get_group_range()[0]);
        if (Index == 0)
          SgSize[0] = SG.get_max_local_range()[0];
      });

  Q.wait();

  bool Fail = false;
  const unsigned int SGSize = static_cast<unsigned int>(SgSize[0]);
  if (SGSize == 0) {
    sycl::free(SyclData, Q);
    sycl::free(SgSize, Q);
    return true;
  }

  const unsigned int NumSg = L / SGSize + ((L % SGSize) ? 1U : 0U);
  for (unsigned int J = 0; J < G; ++J) {
    const unsigned int GroupId = (J % L) / SGSize;
    const unsigned int LocalRange =
        (GroupId + 1 == NumSg) ? (L - GroupId * SGSize) : SGSize;
    Fail |= (SyclData[J].LocalId != ((J % L) % SGSize));
    Fail |= (SyclData[J].LocalRange != LocalRange);
    Fail |= (SyclData[J].MaxLocalRange != SyclData[0].MaxLocalRange);
    Fail |= (SyclData[J].GroupId != GroupId);
    Fail |= (SyclData[J].GroupRange != NumSg);
  }

  sycl::free(SyclData, Q);
  sycl::free(SgSize, Q);
  return Fail;
}

int main() {
  sycl::queue Q;
  bool Fail = false;

  Fail |= check(Q, 240, 80);
  Fail |= check(Q, 8, 4);
  Fail |= check(Q, 24, 12);
  Fail |= check(Q, 1024, 256);

  return Fail;
}
