// This header is used by tests which are interested in __device__ functions
// which appear in a system header.

__device__ int OverloadMe();

namespace ns {
using ::OverloadMe;
}
