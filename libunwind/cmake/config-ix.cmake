include(CMakePushCheckState)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include(CheckSymbolExists)
include(CheckCSourceCompiles)

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

# Check compiler pragmas
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=unknown-pragmas")
  check_c_source_compiles("
#pragma comment(lib, \"c\")
int main() { return 0; }
" LIBUNWIND_HAS_COMMENT_LIB_PRAGMA)
  cmake_pop_check_state()
endif()

# Check compiler flags
check_cxx_compiler_flag(-nostdinc++ LIBUNWIND_HAS_NOSTDINCXX_FLAG)

# Check symbols
check_symbol_exists(__arm__ "" LIBUNWIND_TARGET_ARM)
check_symbol_exists(__USING_SJLJ_EXCEPTIONS__ "" LIBUNWIND_USES_SJLJ_EXCEPTIONS)
check_symbol_exists(__ARM_DWARF_EH__ "" LIBUNWIND_USES_DWARF_EH)

if(LIBUNWIND_TARGET_ARM AND NOT LIBUNWIND_USES_SJLJ_EXCEPTIONS AND NOT LIBUNWIND_USES_DWARF_EH)
  # This condition is copied from __libunwind_config.h
  set(LIBUNWIND_USES_ARM_EHABI ON)
endif()

# Check libraries
if(FUCHSIA)
  set(LIBUNWIND_HAS_DL_LIB NO)
  set(LIBUNWIND_HAS_PTHREAD_LIB NO)
else()
  check_library_exists(dl dladdr "" LIBUNWIND_HAS_DL_LIB)
  check_library_exists(pthread pthread_once "" LIBUNWIND_HAS_PTHREAD_LIB)
endif()
