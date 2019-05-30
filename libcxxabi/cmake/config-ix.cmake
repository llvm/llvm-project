include(CMakePushCheckState)
include(CheckLibraryExists)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckCSourceCompiles)

check_library_exists(c fopen "" LIBCXXABI_HAS_C_LIB)
if (NOT LIBCXXABI_USE_COMPILER_RT)
  check_library_exists(gcc_s __gcc_personality_v0 "" LIBCXXABI_HAS_GCC_S_LIB)
endif ()

# libc++abi is built with -nodefaultlibs, so we want all our checks to also
# use this option, otherwise we may end up with an inconsistency between
# the flags we think we require during configuration (if the checks are
# performed without -nodefaultlibs) and the flags that are actually
# required during compilation (which has the -nodefaultlibs). libc is
# required for the link to go through. We remove sanitizers from the
# configuration checks to avoid spurious link errors.
check_c_compiler_flag(-nodefaultlibs LIBCXXABI_HAS_NODEFAULTLIBS_FLAG)
if (LIBCXXABI_HAS_NODEFAULTLIBS_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nodefaultlibs")
  if (LIBCXXABI_HAS_C_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES c)
  endif ()
  if (LIBCXXABI_USE_COMPILER_RT)
    list(APPEND CMAKE_REQUIRED_FLAGS -rtlib=compiler-rt)
    find_compiler_rt_library(builtins LIBCXXABI_BUILTINS_LIBRARY)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${LIBCXXABI_BUILTINS_LIBRARY}")
  elseif (LIBCXXABI_HAS_GCC_S_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES gcc_s)
  endif ()
  if (MINGW)
    # Mingw64 requires quite a few "C" runtime libraries in order for basic
    # programs to link successfully with -nodefaultlibs.
    if (LIBCXXABI_USE_COMPILER_RT)
      set(MINGW_RUNTIME ${LIBCXXABI_BUILTINS_LIBRARY})
    else ()
      set(MINGW_RUNTIME gcc_s gcc)
    endif()
    set(MINGW_LIBRARIES mingw32 ${MINGW_RUNTIME} moldname mingwex msvcrt advapi32
                        shell32 user32 kernel32 mingw32 ${MINGW_RUNTIME}
                        moldname mingwex msvcrt)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${MINGW_LIBRARIES})
  endif()
  if (CMAKE_C_FLAGS MATCHES -fsanitize OR CMAKE_CXX_FLAGS MATCHES -fsanitize)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-sanitize=all")
  endif ()
  if (CMAKE_C_FLAGS MATCHES -fsanitize-coverage OR CMAKE_CXX_FLAGS MATCHES -fsanitize-coverage)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters")
  endif ()
endif ()

# Check compiler pragmas
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=unknown-pragmas")
  check_c_source_compiles("
#pragma comment(lib, \"c\")
int main() { return 0; }
" LIBCXXABI_HAS_COMMENT_LIB_PRAGMA)
  cmake_pop_check_state()
endif()

# Check compiler flags
check_cxx_compiler_flag(-nostdinc++ LIBCXXABI_HAS_NOSTDINCXX_FLAG)

# Check libraries
check_library_exists(dl dladdr "" LIBCXXABI_HAS_DL_LIB)
check_library_exists(pthread pthread_once "" LIBCXXABI_HAS_PTHREAD_LIB)
check_library_exists(c __cxa_thread_atexit_impl ""
  LIBCXXABI_HAS_CXA_THREAD_ATEXIT_IMPL)
