include(CheckLibraryExists)
include(CheckCXXCompilerFlag)
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
check_library_exists(c fopen "" LIBCXX_HAS_C_LIB)
check_library_exists(m ccos "" LIBCXX_HAS_M_LIB)
check_library_exists(rt clock_gettime "" LIBCXX_HAS_RT_LIB)
check_library_exists(gcc_s __gcc_personality_v0 "" LIBCXX_HAS_GCC_S_LIB)
