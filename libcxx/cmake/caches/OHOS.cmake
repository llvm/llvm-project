
# include(Platform/Linux)

# OHOS has soname, but binary names must end in ".so" so we cannot append
# a version number.  Also we cannot portably represent symlinks on the host.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

# OHOS reportedly ignores RPATH, and we cannot predict the install
# location anyway.
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "")

set(LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
set(CMAKE_CXX_COMPILER_TARGET "aarch64-linux-gnu" CACHE STRING "")
