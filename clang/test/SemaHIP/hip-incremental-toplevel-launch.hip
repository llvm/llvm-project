// In incremental mode (clang-repl) statements may appear at the top level, so a
// HIP kernel launch can have no enclosing caller function. During device
// compilation such a launch must be accepted (treated as a host-side launch)
// rather than rejected as an unsupported device-side kernel launch.

// RUN: %clang_cc1 -fsyntax-only -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -fincremental-extensions -verify %s
// expected-no-diagnostics

#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))

typedef unsigned long size_t;

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1)
      : x(x), y(y), z(z) {}
};

typedef struct hipStream *hipStream_t;
typedef enum hipError {} hipError_t;
int hipConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0,
                     hipStream_t stream = 0);

__global__ void kernel() {}

// Top-level kernel launch with no surrounding function, i.e. a null caller.
kernel<<<1, 1>>>();
