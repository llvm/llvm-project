#ifndef CSAN_TEST_RACE_H
#define CSAN_TEST_RACE_H

#if defined(__HIP__)
extern "C" __device__ unsigned long long __csan_get_num_data_races(void);
#  define CSAN_DEVICE __device__
#elif defined(__AMDGCN__)
extern "C" unsigned long long __csan_get_num_data_races(void);
#  define CSAN_DEVICE
#else
extern "C" unsigned long long __csan_get_num_data_races(void);
#  define CSAN_DEVICE
#endif

CSAN_DEVICE static inline int race_found(void) noexcept {
  return __csan_get_num_data_races() != 0;
}

#define RACE_MAX_ITERS (1 << 20)
#define RACE_UNTIL_FOUND(i)                                                    \
  for (int i = 0; i < RACE_MAX_ITERS && !race_found(); ++i)

#if defined(__HIP_DEVICE_COMPILE__)
#  include <gpuintrin.h>
#  define CSAN_TID()                                                           \
    (__gpu_num_threads(__GPU_X_DIM) * __gpu_block_id(__GPU_X_DIM) +            \
     __gpu_thread_id(__GPU_X_DIM))
#else
#  define CSAN_TID() 0u
#endif

#endif
