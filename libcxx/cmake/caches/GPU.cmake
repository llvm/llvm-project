# Handle default arguments when done through the LLVM runtimes interface.
foreach(target amdgcn-amd-amdhsa nvptx64-nvidia-cuda)
  set(RUNTIMES_${target}_LIBCXX_ABI_VERSION 2 CACHE STRING "")
  set(RUNTIMES_${target}_LIBCXX_CXX_ABI none CACHE STRING "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_STATIC ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_RANDOM_DEVICE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_LOCALIZATION OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_UNICODE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_WIDE_CHARACTERS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_HAS_TERMINAL_AVAILABLE OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_RTTI OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_STATIC_ABI_LIBRARY ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_THREADS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_MONOTONIC_CLOCK ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_INSTALL_LIBRARY ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_LIBC "llvm-libc" CACHE STRING "")
  set(RUNTIMES_${target}_LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS ON CACHE BOOL "")

  # Configuration options for libcxxabi.
  set(RUNTIMES_${target}_LIBCXXABI_BAREMETAL ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_ENABLE_THREADS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS OFF CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_USE_LLVM_UNWINDER OFF CACHE BOOL "")

  # Target specific compile flags.
  if(${target} MATCHES "^amdgcn")
    set(RUNTIMES_${target}_LIBCXX_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;-Xclang;-mcode-object-version=none" CACHE STRING "")
    set(RUNTIMES_${target}_LIBCXXABI_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;-Xclang;-mcode-object-version=none" CACHE STRING "")
    set(RUNTIMES_${target}_CMAKE_REQUIRED_FLAGS "-nogpulib -nodefaultlibs" CACHE STRING "")
  else()
    set(RUNTIMES_${target}_LIBCXX_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;--cuda-feature=+ptx63" CACHE STRING "")
    set(RUNTIMES_${target}_LIBCXXABI_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;--cuda-feature=+ptx63" CACHE STRING "")
    set(RUNTIMES_${target}_CMAKE_REQUIRED_FLAGS
        "-flto -nodefaultlibs -c -Wno-unused-command-line-argument" CACHE STRING "")
  endif()
endforeach()

# Handle default arguments when being built directly.
if(${CMAKE_CXX_COMPILER_TARGET} MATCHES "^amdgcn|^nvptx")
  set(LIBCXX_ABI_VERSION 2 CACHE STRING "")
  set(LIBCXX_CXX_ABI none CACHE STRING "")
  set(LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_STATIC ON CACHE BOOL "")
  set(LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_RANDOM_DEVICE OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_LOCALIZATION OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_UNICODE OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_WIDE_CHARACTERS OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
  set(LIBCXX_HAS_TERMINAL_AVAILABLE OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_RTTI OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_STATIC_ABI_LIBRARY ON CACHE BOOL "")
  set(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE BOOL "")
  set(LIBCXX_ENABLE_THREADS OFF CACHE BOOL "")
  set(LIBCXX_ENABLE_MONOTONIC_CLOCK ON CACHE BOOL "")
  set(LIBCXX_INSTALL_LIBRARY ON CACHE BOOL "")
  set(LIBCXX_LIBC "llvm-libc" CACHE STRING "")
  set(LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
  set(LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS ON CACHE BOOL "")

  # Configuration options for libcxxabi.
  set(LIBCXXABI_BAREMETAL ON CACHE BOOL "")
  set(LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
  set(LIBCXXABI_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
  set(LIBCXXABI_ENABLE_THREADS OFF CACHE BOOL "")
  set(LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS OFF CACHE BOOL "")
  set(LIBCXXABI_USE_LLVM_UNWINDER OFF CACHE BOOL "")

  # Target specific compile flags.
  if(${target} MATCHES "^amdgcn")
    set(LIBCXX_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;-Xclang;-mcode-object-version=none" CACHE STRING "")
    set(LIBCXXABI_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;-Xclang;-mcode-object-version=none" CACHE STRING "")
    set(CMAKE_REQUIRED_FLAGS "-nogpulib" CACHE STRING "")
  else()
    set(LIBCXX_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;--cuda-feature=+ptx63" CACHE STRING "")
    set(LIBCXXABI_ADDITIONAL_COMPILE_FLAGS
        "-nogpulib;-flto;-fconvergent-functions;--cuda-feature=+ptx63" CACHE STRING "")
    set(CMAKE_REQUIRED_FLAGS "-flto;-c;-Wno-unused-command-line-argument" CACHE STRING "")
  endif()
endif()
