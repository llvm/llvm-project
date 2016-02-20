INCLUDE(CheckCXXSourceCompiles)

# Sometimes linking against libatomic is required for atomic ops, if
# the platform doesn't support lock-free atomics.
#
# We could modify LLVM's CheckAtomic module and have it check for 64-bit
# atomics instead. However, we would like to avoid careless uses of 64-bit
# atomics inside LLVM over time on 32-bit platforms.

function(check_cxx_atomics varname)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "-std=c++11 -nostdinc++ -isystem ${LIBCXX_SOURCE_DIR}/include")
  if (${LIBCXX_GCC_TOOLCHAIN})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} --gcc-toolchain=${LIBCXX_GCC_TOOLCHAIN}")
  endif()
  check_cxx_source_compiles("
#include <cstdint>
#include <atomic>
std::atomic<uintptr_t> x;
std::atomic<uintmax_t> y;
int main() {
  return x + y;
}
" ${varname})
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endfunction(check_cxx_atomics)

check_cxx_atomics(LIBCXX_HAVE_CXX_ATOMICS_WITHOUT_LIB)
# If not, check if the library exists, and atomics work with it.
if(NOT LIBCXX_HAVE_CXX_ATOMICS_WITHOUT_LIB)
  check_library_exists(atomic __atomic_fetch_add_8 "" LIBCXX_HAS_ATOMIC_LIB)
  if(LIBCXX_HAS_ATOMIC_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
    check_cxx_atomics(LIBCXX_HAVE_CXX_ATOMICS_WITH_LIB)
    if (NOT LIBCXX_HAVE_CXX_ATOMICS_WITH_LIB)
      message(WARNING "Host compiler must support std::atomic!")
    endif()
  else()
    message(WARNING "Host compiler appears to require libatomic, but cannot find it.")
  endif()
endif()
