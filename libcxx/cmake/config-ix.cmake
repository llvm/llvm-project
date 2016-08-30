include(CheckLibraryExists)
include(CheckCXXCompilerFlag)

check_library_exists(c fopen "" LIBCXX_HAS_C_LIB)
if (NOT LIBCXX_USE_COMPILER_RT)
  check_library_exists(gcc_s __gcc_personality_v0 "" LIBCXX_HAS_GCC_S_LIB)
endif()

# libc++ is built with -nodefaultlibs, so we want all our checks to also
# use this option, otherwise we may end up with an inconsistency between
# the flags we think we require during configuration (if the checks are
# performed without -nodefaultlibs) and the flags that are actually
# required during compilation (which has the -nodefaultlibs). libc is
# required for the link to go through. We remove sanitizers from the
# configuration checks to avoid spurious link errors.
check_cxx_compiler_flag(-nodefaultlibs LIBCXX_SUPPORTS_NODEFAULTLIBS_FLAG)
if (LIBCXX_SUPPORTS_NODEFAULTLIBS_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nodefaultlibs")
  if (LIBCXX_HAS_C_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES c)
  endif ()
  if (LIBCXX_USE_COMPILER_RT)
    list(APPEND CMAKE_REQUIRED_LIBRARIES -rtlib=compiler-rt)
  elseif (LIBCXX_HAS_GCC_S_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES gcc_s)
  endif ()
  if (CMAKE_C_FLAGS MATCHES -fsanitize OR CMAKE_CXX_FLAGS MATCHES -fsanitize)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-sanitize=all")
  endif ()
endif ()

include(CheckLibcxxAtomic)

# Check compiler flags

check_cxx_compiler_flag(/WX                     LIBCXX_HAS_WX_FLAG)
check_cxx_compiler_flag(/WX-                    LIBCXX_HAS_NO_WX_FLAG)
check_cxx_compiler_flag(/EHsc                   LIBCXX_HAS_EHSC_FLAG)
check_cxx_compiler_flag(/EHs-                   LIBCXX_HAS_NO_EHS_FLAG)
check_cxx_compiler_flag(/EHa-                   LIBCXX_HAS_NO_EHA_FLAG)
check_cxx_compiler_flag(/GR-                    LIBCXX_HAS_NO_GR_FLAG)


# Check libraries
check_library_exists(pthread pthread_create "" LIBCXX_HAS_PTHREAD_LIB)
check_library_exists(m ccos "" LIBCXX_HAS_M_LIB)
check_library_exists(rt clock_gettime "" LIBCXX_HAS_RT_LIB)
