
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckLibraryExists)

check_library_exists(c fopen "" LIBUNWIND_HAS_C_LIB)

if (NOT LIBUNWIND_USE_COMPILER_RT)
  check_library_exists(gcc_s __gcc_personality_v0 "" LIBUNWIND_HAS_GCC_S_LIB)
  check_library_exists(gcc __absvdi2 "" LIBUNWIND_HAS_GCC_LIB)
endif()

# libunwind is built with -nodefaultlibs, so we want all our checks to also
# use this option, otherwise we may end up with an inconsistency between
# the flags we think we require during configuration (if the checks are
# performed without -nodefaultlibs) and the flags that are actually
# required during compilation (which has the -nodefaultlibs). libc is
# required for the link to go through. We remove sanitizers from the
# configuration checks to avoid spurious link errors.
check_c_compiler_flag(-nodefaultlibs LIBUNWIND_HAS_NODEFAULTLIBS_FLAG)
if (LIBUNWIND_HAS_NODEFAULTLIBS_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nodefaultlibs")
  if (LIBUNWIND_HAS_C_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES c)
  endif ()
  if (LIBUNWIND_USE_COMPILER_RT)
    find_compiler_rt_library(builtins LIBUNWIND_BUILTINS_LIBRARY)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${LIBUNWIND_BUILTINS_LIBRARY}")
  else ()
    if (LIBUNWIND_HAS_GCC_S_LIB)
      list(APPEND CMAKE_REQUIRED_LIBRARIES gcc_s)
    endif ()
    if (LIBUNWIND_HAS_GCC_LIB)
      list(APPEND CMAKE_REQUIRED_LIBRARIES gcc)
    endif ()
  endif ()
  if (MINGW)
    # Mingw64 requires quite a few "C" runtime libraries in order for basic
    # programs to link successfully with -nodefaultlibs.
    if (LIBUNWIND_USE_COMPILER_RT)
      set(MINGW_RUNTIME ${LIBUNWIND_BUILTINS_LIBRARY})
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

# Check compiler flags
check_c_compiler_flag(-funwind-tables         LIBUNWIND_HAS_FUNWIND_TABLES)
check_cxx_compiler_flag(-fno-exceptions       LIBUNWIND_HAS_NO_EXCEPTIONS_FLAG)
check_cxx_compiler_flag(-fno-rtti             LIBUNWIND_HAS_NO_RTTI_FLAG)
check_cxx_compiler_flag(-fstrict-aliasing     LIBUNWIND_HAS_FSTRICT_ALIASING_FLAG)
check_cxx_compiler_flag(-nostdinc++           LIBUNWIND_HAS_NOSTDINCXX_FLAG)
check_cxx_compiler_flag(-Wall                 LIBUNWIND_HAS_WALL_FLAG)
check_cxx_compiler_flag(-W                    LIBUNWIND_HAS_W_FLAG)
check_cxx_compiler_flag(-Wno-unused-function  LIBUNWIND_HAS_WNO_UNUSED_FUNCTION_FLAG)
check_cxx_compiler_flag(-Wunused-variable     LIBUNWIND_HAS_WUNUSED_VARIABLE_FLAG)
check_cxx_compiler_flag(-Wunused-parameter    LIBUNWIND_HAS_WUNUSED_PARAMETER_FLAG)
check_cxx_compiler_flag(-Wstrict-aliasing     LIBUNWIND_HAS_WSTRICT_ALIASING_FLAG)
check_cxx_compiler_flag(-Wstrict-overflow     LIBUNWIND_HAS_WSTRICT_OVERFLOW_FLAG)
check_cxx_compiler_flag(-Wwrite-strings       LIBUNWIND_HAS_WWRITE_STRINGS_FLAG)
check_cxx_compiler_flag(-Wchar-subscripts     LIBUNWIND_HAS_WCHAR_SUBSCRIPTS_FLAG)
check_cxx_compiler_flag(-Wmismatched-tags     LIBUNWIND_HAS_WMISMATCHED_TAGS_FLAG)
check_cxx_compiler_flag(-Wmissing-braces      LIBUNWIND_HAS_WMISSING_BRACES_FLAG)
check_cxx_compiler_flag(-Wshorten-64-to-32    LIBUNWIND_HAS_WSHORTEN_64_TO_32_FLAG)
check_cxx_compiler_flag(-Wsign-conversion     LIBUNWIND_HAS_WSIGN_CONVERSION_FLAG)
check_cxx_compiler_flag(-Wsign-compare        LIBUNWIND_HAS_WSIGN_COMPARE_FLAG)
check_cxx_compiler_flag(-Wshadow              LIBUNWIND_HAS_WSHADOW_FLAG)
check_cxx_compiler_flag(-Wconversion          LIBUNWIND_HAS_WCONVERSION_FLAG)
check_cxx_compiler_flag(-Wnewline-eof         LIBUNWIND_HAS_WNEWLINE_EOF_FLAG)
check_cxx_compiler_flag(-Wundef               LIBUNWIND_HAS_WUNDEF_FLAG)
check_cxx_compiler_flag(-pedantic             LIBUNWIND_HAS_PEDANTIC_FLAG)
check_cxx_compiler_flag(-Werror               LIBUNWIND_HAS_WERROR_FLAG)
check_cxx_compiler_flag(-Wno-error            LIBUNWIND_HAS_WNO_ERROR_FLAG)
check_cxx_compiler_flag(/WX                   LIBUNWIND_HAS_WX_FLAG)
check_cxx_compiler_flag(/WX-                  LIBUNWIND_HAS_NO_WX_FLAG)
check_cxx_compiler_flag(/EHsc                 LIBUNWIND_HAS_EHSC_FLAG)
check_cxx_compiler_flag(/EHs-                 LIBUNWIND_HAS_NO_EHS_FLAG)
check_cxx_compiler_flag(/EHa-                 LIBUNWIND_HAS_NO_EHA_FLAG)
check_cxx_compiler_flag(/GR-                  LIBUNWIND_HAS_NO_GR_FLAG)
check_cxx_compiler_flag(-std=c++11            LIBUNWIND_HAS_STD_CXX11)

if(LIBUNWIND_HAS_STD_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

check_library_exists(dl dladdr "" LIBUNWIND_HAS_DL_LIB)
check_library_exists(pthread pthread_once "" LIBUNWIND_HAS_PTHREAD_LIB)

