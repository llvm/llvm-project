
CHECK_CXX_SOURCE_COMPILES("
#ifdef _MSC_VER
#include <Intrin.h> /* Workaround for PR19898. */
#include <windows.h>
#endif
int main() {
#ifdef _MSC_VER
        volatile LONG val = 1;
        MemoryBarrier();
        InterlockedCompareExchange(&val, 0, 1);
        InterlockedIncrement(&val);
        InterlockedDecrement(&val);
#else
        volatile unsigned long val = 1;
        __sync_synchronize();
        __sync_val_compare_and_swap(&val, 1, 0);
        __sync_add_and_fetch(&val, 1);
        __sync_sub_and_fetch(&val, 1);
#endif
        return 0;
      }
" COMPILER_RT_TARGET_HAS_ATOMICS)

add_custom_target(profile)

set(PROFILE_SOURCES
  GCDAProfiling.c
  InstrProfiling.c
  InstrProfilingValue.c
  InstrProfilingBuffer.c
  InstrProfilingFile.c
  InstrProfilingMerge.c
  InstrProfilingMergeFile.c
  InstrProfilingWriter.c
  InstrProfilingPlatformDarwin.c
  InstrProfilingPlatformLinux.c
  InstrProfilingPlatformOther.c
  InstrProfilingRuntime.cc
  InstrProfilingUtil.c)

if(WIN32)
    list(APPEND PROFILE_SOURCES WindowsMMap.c)
endif()

if(UNIX)
 set(EXTRA_FLAGS
     -fPIC
     -Wno-pedantic)
endif()

if(COMPILER_RT_TARGET_HAS_ATOMICS)
 set(EXTRA_FLAGS
     ${EXTRA_FLAGS}
     -DCOMPILER_RT_HAS_ATOMICS=1)
endif() 

if(APPLE)
  add_compiler_rt_runtime(clang_rt.profile
    STATIC
    OS ${PROFILE_SUPPORTED_OS}
    ARCHS ${PROFILE_SUPPORTED_ARCH}
    SOURCES ${PROFILE_SOURCES}
    PARENT_TARGET profile)
else()
  add_compiler_rt_runtime(clang_rt.profile
    STATIC
    ARCHS ${PROFILE_SUPPORTED_ARCH}
    CFLAGS ${EXTRA_FLAGS}
    SOURCES ${PROFILE_SOURCES}
    PARENT_TARGET profile)
endif()

add_dependencies(compiler-rt profile)
