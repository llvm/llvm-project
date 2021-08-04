# CMake build for CompilerRT.
#
# An important constraint of the build is that it only produces libraries
# based on the ability of the host toolchain to target various platforms.

cmake_minimum_required(VERSION 3.13.4)

# Check if compiler-rt is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR COMPILER_RT_STANDALONE_BUILD)
  project(CompilerRT C CXX ASM)
  set(COMPILER_RT_STANDALONE_BUILD TRUE)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# Add path for custom compiler-rt modules.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CFG_RESOLVED_INTDIR "${CMAKE_CFG_INTDIR}/")
else()
  set(CMAKE_CFG_RESOLVED_INTDIR "")
endif()

include(base-config-ix)
include(CompilerRTUtils)

option(COMPILER_RT_BUILD_BUILTINS "Build builtins" ON)
mark_as_advanced(COMPILER_RT_BUILD_BUILTINS)
option(COMPILER_RT_BUILD_CRT "Build crtbegin.o/crtend.o" ON)
mark_as_advanced(COMPILER_RT_BUILD_CRT)
option(COMPILER_RT_CRT_USE_EH_FRAME_REGISTRY "Use eh_frame in crtbegin.o/crtend.o" ON)
mark_as_advanced(COMPILER_RT_CRT_USE_EH_FRAME_REGISTRY)
option(COMPILER_RT_BUILD_SANITIZERS "Build sanitizers" ON)
mark_as_advanced(COMPILER_RT_BUILD_SANITIZERS)
option(COMPILER_RT_BUILD_XRAY "Build xray" ON)
mark_as_advanced(COMPILER_RT_BUILD_XRAY)
option(COMPILER_RT_BUILD_LIBFUZZER "Build libFuzzer" ON)
mark_as_advanced(COMPILER_RT_BUILD_LIBFUZZER)
option(COMPILER_RT_BUILD_PROFILE "Build profile runtime" ON)
mark_as_advanced(COMPILER_RT_BUILD_PROFILE)
option(COMPILER_RT_BUILD_MEMPROF "Build memory profiling runtime" ON)
mark_as_advanced(COMPILER_RT_BUILD_MEMPROF)
option(COMPILER_RT_BUILD_XRAY_NO_PREINIT "Build xray with no preinit patching" OFF)
mark_as_advanced(COMPILER_RT_BUILD_XRAY_NO_PREINIT)
option(COMPILER_RT_BUILD_ORC "Build ORC runtime" ON)
mark_as_advanced(COMPILER_RT_BUILD_ORC)

set(COMPILER_RT_ASAN_SHADOW_SCALE ""
    CACHE STRING "Override the shadow scale to be used in ASan runtime")

if (NOT COMPILER_RT_ASAN_SHADOW_SCALE STREQUAL "")
  # Check that the shadow scale value is valid.
  if (NOT (COMPILER_RT_ASAN_SHADOW_SCALE GREATER -1 AND
           COMPILER_RT_ASAN_SHADOW_SCALE LESS 8))
    message(FATAL_ERROR "
      Invalid ASan Shadow Scale '${COMPILER_RT_ASAN_SHADOW_SCALE}'.")
  endif()

  set(COMPILER_RT_ASAN_SHADOW_SCALE_LLVM_FLAG
      -mllvm -asan-mapping-scale=${COMPILER_RT_ASAN_SHADOW_SCALE})
  set(COMPILER_RT_ASAN_SHADOW_SCALE_DEFINITION
      ASAN_SHADOW_SCALE=${COMPILER_RT_ASAN_SHADOW_SCALE})
  set(COMPILER_RT_ASAN_SHADOW_SCALE_FLAG
      -D${COMPILER_RT_ASAN_SHADOW_SCALE_DEFINITION})
endif()

if(FUCHSIA)
  set(COMPILER_RT_HWASAN_WITH_INTERCEPTORS_DEFAULT OFF)
else()
  set(COMPILER_RT_HWASAN_WITH_INTERCEPTORS_DEFAULT ON)
endif()
set(COMPILER_RT_HWASAN_WITH_INTERCEPTORS ${COMPILER_RT_HWASAN_WITH_INTERCEPTORS_DEFAULT} CACHE BOOL "Enable libc interceptors in HWASan (testing mode)")

set(COMPILER_RT_BAREMETAL_BUILD OFF CACHE BOOL
  "Build for a bare-metal target.")

if (COMPILER_RT_STANDALONE_BUILD)
  load_llvm_config()
  if (TARGET intrinsics_gen)
    # Loading the llvm config causes this target to be imported so place it
    # under the appropriate folder in an IDE.
    set_target_properties(intrinsics_gen PROPERTIES FOLDER "Compiler-RT Misc")
  endif()

  find_package(Python3 COMPONENTS Interpreter)
  if(NOT Python3_Interpreter_FOUND)
    message(WARNING "Python3 not found, using python2 as a fallback")
    find_package(Python2 COMPONENTS Interpreter REQUIRED)
    if(Python2_VERSION VERSION_LESS 2.7)
      message(SEND_ERROR "Python 2.7 or newer is required")
    endif()

    # Treat python2 as python3
    add_executable(Python3::Interpreter IMPORTED)
    set_target_properties(Python3::Interpreter PROPERTIES
      IMPORTED_LOCATION ${Python2_EXECUTABLE})
    set(Python3_EXECUTABLE ${Python2_EXECUTABLE})
  endif()

  # Ensure that fat libraries are built correctly on Darwin
  if(APPLE)
    include(UseLibtool)
  endif()

  # Define default arguments to lit.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
  set(LLVM_LIT_OUTPUT_DIR "${COMPILER_RT_EXEC_OUTPUT_DIR}")
endif()

construct_compiler_rt_default_triple()
if ("${COMPILER_RT_DEFAULT_TARGET_TRIPLE}" MATCHES ".*hf$")
  if (${COMPILER_RT_DEFAULT_TARGET_ARCH} MATCHES "^arm")
    set(COMPILER_RT_DEFAULT_TARGET_ARCH "armhf")
    CHECK_SYMBOL_EXISTS (__thumb__ "" COMPILER_RT_ARM_THUMB)
  endif()
endif()
if ("${COMPILER_RT_DEFAULT_TARGET_TRIPLE}" MATCHES ".*android.*")
  set(ANDROID 1)
  string(REGEX MATCH "-target(=| +)[^ ]+android[a-z]*([0-9]+)" ANDROID_API_LEVEL "${CMAKE_C_FLAGS}")
  set(ANDROID_API_LEVEL ${CMAKE_MATCH_2})
endif()
pythonize_bool(ANDROID)

set(COMPILER_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPILER_RT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

pythonize_bool(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)

# We support running instrumented tests when we're not cross compiling
# and target a UNIX-like system or Windows.
# We can run tests on Android even when we are cross-compiling.
if(("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND (UNIX OR WIN32)) OR ANDROID
   OR COMPILER_RT_EMULATOR)
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" ON)
else()
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" OFF)
endif()

option(COMPILER_RT_DEBUG "Build runtimes with full debug info" OFF)
option(COMPILER_RT_EXTERNALIZE_DEBUGINFO
  "Generate dSYM files and strip executables and libraries (Darwin Only)" OFF)
# COMPILER_RT_DEBUG_PYBOOL is used by lit.common.configured.in.
pythonize_bool(COMPILER_RT_DEBUG)

option(COMPILER_RT_INTERCEPT_LIBDISPATCH
  "Support interception of libdispatch (GCD). Requires '-fblocks'" OFF)
option(COMPILER_RT_LIBDISPATCH_INSTALL_PATH
  "Specify if libdispatch is installed in a custom location" "")
if (COMPILER_RT_INTERCEPT_LIBDISPATCH AND NOT APPLE)
  set(COMPILER_RT_LIBDISPATCH_CFLAGS -fblocks)
  set(COMPILER_RT_TEST_LIBDISPATCH_CFLAGS)
  if (COMPILER_RT_LIBDISPATCH_INSTALL_PATH)
    list(APPEND COMPILER_RT_TEST_LIBDISPATCH_CFLAGS
        -I${COMPILER_RT_LIBDISPATCH_INSTALL_PATH}/include
        -L${COMPILER_RT_LIBDISPATCH_INSTALL_PATH}/lib
        -Wl,-rpath=${COMPILER_RT_LIBDISPATCH_INSTALL_PATH}/lib)
  endif()
  list(APPEND COMPILER_RT_TEST_LIBDISPATCH_CFLAGS -lBlocksRuntime -ldispatch)
endif()
if (APPLE) # Always enable on Apple platforms.
  set(COMPILER_RT_INTERCEPT_LIBDISPATCH ON)
endif()
pythonize_bool(COMPILER_RT_INTERCEPT_LIBDISPATCH)

if(APPLE AND SANITIZER_MIN_OSX_VERSION AND SANITIZER_MIN_OSX_VERSION VERSION_LESS "10.9")
  # Mac OS X prior to 10.9 had problems with exporting symbols from
  # libc++/libc++abi.
  set(cxxabi_supported OFF)
else()
  set(cxxabi_supported ON)
endif()

option(SANITIZER_ALLOW_CXXABI "Allow use of C++ ABI details in ubsan" ON)

set(SANITIZER_CAN_USE_CXXABI OFF)
if (cxxabi_supported AND SANITIZER_ALLOW_CXXABI)
  set(SANITIZER_CAN_USE_CXXABI ON)
endif()
pythonize_bool(SANITIZER_CAN_USE_CXXABI)

macro(handle_default_cxx_lib var)
  # Specifying -stdlib= in CMAKE_CXX_FLAGS overrides the defaults.
  if (CMAKE_CXX_FLAGS MATCHES "-stdlib=([a-zA-Z+]*)")
    set(${var}_LIBNAME "${CMAKE_MATCH_1}")
    set(${var}_SYSTEM 1)
  elseif (${var} STREQUAL "default")
    if (APPLE OR CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
      set(${var}_LIBNAME "libc++")
      set(${var}_SYSTEM 1)
    elseif (FUCHSIA)
      set(${var}_LIBNAME "libc++")
      set(${var}_INTREE 1)
    else()
      set(${var}_LIBNAME "libstdc++")
      set(${var}_SYSTEM 1)
    endif()
  else()
    set(${var}_LIBNAME "${${var}}")
    set(${var}_SYSTEM 1)
  endif()
endmacro()

# This is either directly the C++ ABI library or the full C++ library
# which pulls in the ABI transitively.
set(SANITIZER_CXX_ABI "default" CACHE STRING
    "Specify C++ ABI library to use.")
set(CXXABIS none default libstdc++ libc++ libcxxabi)
set_property(CACHE SANITIZER_CXX_ABI PROPERTY STRINGS ;${CXXABIS})
handle_default_cxx_lib(SANITIZER_CXX_ABI)

# This needs to be a full C++ library for linking gtest and unit tests.
set(SANITIZER_TEST_CXX "default" CACHE STRING
    "Specify C++ library to use for tests.")
set(CXXLIBS none default libstdc++ libc++)
set_property(CACHE SANITIZER_TEST_CXX PROPERTY STRINGS ;${CXXLIBS})
handle_default_cxx_lib(SANITIZER_TEST_CXX)

set(DEFAULT_SANITIZER_USE_STATIC_LLVM_UNWINDER OFF)
if (FUCHSIA)
  set(DEFAULT_SANITIZER_USE_STATIC_LLVM_UNWINDER ON)
elseif (DEFINED LIBUNWIND_ENABLE_SHARED AND NOT LIBUNWIND_ENABLE_SHARED)
  set(DEFAULT_SANITIZER_USE_STATIC_LLVM_UNWINDER ON)
endif()

option(SANITIZER_USE_STATIC_LLVM_UNWINDER
  "Use static LLVM unwinder." ${DEFAULT_SANITIZER_USE_STATIC_LLVM_UNWINDER})

set(DEFAULT_SANITIZER_USE_STATIC_CXX_ABI OFF)
if (DEFINED LIBCXXABI_ENABLE_SHARED AND NOT LIBCXXABI_ENABLE_SHARED)
  set(DEFAULT_SANITIZER_USE_STATIC_CXX_ABI ON)
endif()

option(SANITIZER_USE_STATIC_CXX_ABI
  "Use static libc++abi." ${DEFAULT_SANITIZER_USE_STATIC_CXX_ABI})

set(DEFAULT_COMPILER_RT_USE_BUILTINS_LIBRARY OFF)
if (FUCHSIA)
  set(DEFAULT_COMPILER_RT_USE_BUILTINS_LIBRARY ON)
endif()

option(COMPILER_RT_USE_BUILTINS_LIBRARY
  "Use compiler-rt builtins instead of libgcc" ${DEFAULT_COMPILER_RT_USE_BUILTINS_LIBRARY})

include(config-ix)

#================================
# Setup Compiler Flags
#================================

if(MSVC)
  # Override any existing /W flags with /W4. This is what LLVM does.  Failing to
  # remove other /W[0-4] flags will result in a warning about overriding a
  # previous flag.
  if (COMPILER_RT_HAS_W4_FLAG)
    string(REGEX REPLACE " /W[0-4]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE " /W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    append_string_if(COMPILER_RT_HAS_W4_FLAG /W4 CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
else()
  append_string_if(COMPILER_RT_HAS_WALL_FLAG -Wall CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()
if(COMPILER_RT_ENABLE_WERROR)
  append_string_if(COMPILER_RT_HAS_WERROR_FLAG -Werror CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append_string_if(COMPILER_RT_HAS_WX_FLAG /WX CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

append_string_if(COMPILER_RT_HAS_STD_CXX14_FLAG -std=c++14 CMAKE_CXX_FLAGS)

# Emulate C99 and C++11's __func__ for MSVC prior to 2013 CTP.
if(NOT COMPILER_RT_HAS_FUNC_SYMBOL)
  add_definitions(-D__func__=__FUNCTION__)
endif()

# Provide some common commandline flags for Sanitizer runtimes.
if("${ANDROID_API_LEVEL}" GREATER_EQUAL 29)
  list(APPEND SANITIZER_COMMON_CFLAGS -fno-emulated-tls)
  string(APPEND COMPILER_RT_TEST_COMPILER_CFLAGS " -fno-emulated-tls")
endif()
if(NOT WIN32)
  append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC SANITIZER_COMMON_CFLAGS)
endif()
append_list_if(COMPILER_RT_HAS_FNO_BUILTIN_FLAG -fno-builtin SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions SANITIZER_COMMON_CFLAGS)
if(NOT COMPILER_RT_DEBUG AND NOT APPLE)
  append_list_if(COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG -fomit-frame-pointer SANITIZER_COMMON_CFLAGS)
endif()
append_list_if(COMPILER_RT_HAS_FUNWIND_TABLES_FLAG -funwind-tables SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG -fno-stack-protector SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_SANITIZE_SAFE_STACK_FLAG -fno-sanitize=safe-stack SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden SANITIZER_COMMON_CFLAGS)
if(NOT COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG)
  append_list_if(COMPILER_RT_HAS_FVISIBILITY_INLINES_HIDDEN_FLAG -fvisibility-inlines-hidden SANITIZER_COMMON_CFLAGS)
endif()
append_list_if(COMPILER_RT_HAS_FNO_LTO_FLAG -fno-lto SANITIZER_COMMON_CFLAGS)

# By default do not instrument or use profdata for compiler-rt.
if(NOT COMPILER_RT_ENABLE_PGO)
  if(LLVM_PROFDATA_FILE AND COMPILER_RT_HAS_FNO_PROFILE_INSTR_USE_FLAG)
    list(APPEND SANITIZER_COMMON_CFLAGS "-fno-profile-instr-use")
  endif()
  if(LLVM_BUILD_INSTRUMENTED MATCHES IR AND COMPILER_RT_HAS_FNO_PROFILE_GENERATE_FLAG)
    list(APPEND SANITIZER_COMMON_CFLAGS "-fno-profile-generate")
  elseif(LLVM_BUILD_INSTRUMENTED AND COMPILER_RT_HAS_FNO_PROFILE_INSTR_GENERATE_FLAG)
    list(APPEND SANITIZER_COMMON_CFLAGS "-fno-profile-instr-generate")
  endif()
endif()

# The following is a workaround for powerpc64le. This is the only architecture
# that requires -fno-function-sections to work properly. If lacking, the ASan
# Linux test function-sections-are-bad.cpp fails with the following error:
# 'undefined symbol: __sanitizer_unaligned_load32'.
if(DEFINED TARGET_powerpc64le_CFLAGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "XL")
    append("-qnofuncsect" TARGET_powerpc64le_CFLAGS)
  else()
    append_list_if(COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG -fno-function-sections TARGET_powerpc64le_CFLAGS)
  endif()
endif()

# The following is a workaround for s390x.  This avoids creation of "partial
# inline" function fragments when building the asan libraries with certain
# GCC versions.  The presence of those fragments, in particular for the
# interceptors, changes backtraces seen in asan error cases, which causes
# testsuite failures.
if("${COMPILER_RT_DEFAULT_TARGET_ARCH}" MATCHES "s390x")
  append_list_if(COMPILER_RT_HAS_FNO_PARTIAL_INLINING_FLAG -fno-partial-inlining SANITIZER_COMMON_CFLAGS)
endif()

if(MSVC)
  # Replace the /M[DT][d] flags with /MT, and strip any definitions of _DEBUG,
  # which cause definition mismatches at link time.
  # FIXME: In fact, sanitizers should support both /MT and /MD, see PR20214.
  if(COMPILER_RT_HAS_MT_FLAG)
    foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      string(REGEX REPLACE "/M[DT]d" "/MT" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/D_DEBUG" "" ${flag_var} "${${flag_var}}")
    endforeach()
  endif()
  append_list_if(COMPILER_RT_HAS_Oy_FLAG /Oy- SANITIZER_COMMON_CFLAGS)
  append_list_if(COMPILER_RT_HAS_GS_FLAG /GS- SANITIZER_COMMON_CFLAGS)

  # Disable thread safe initialization for static locals. ASan shouldn't need
  # it. Thread safe initialization assumes that the CRT has already been
  # initialized, but ASan initializes before the CRT.
  list(APPEND SANITIZER_COMMON_CFLAGS /Zc:threadSafeInit-)
endif()

append_list_if(COMPILER_RT_DEBUG -DSANITIZER_DEBUG=1 SANITIZER_COMMON_CFLAGS)

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  list(APPEND THREAD_SAFETY_FLAGS
      "-Werror=thread-safety"
      "-Werror=thread-safety-reference"
      "-Werror=thread-safety-beta"
  )
  list(APPEND SANITIZER_COMMON_CFLAGS ${THREAD_SAFETY_FLAGS})
  string(REPLACE ";" " " thread_safety_flags_space_sep "${THREAD_SAFETY_FLAGS}")
  string(APPEND COMPILER_RT_TEST_COMPILER_CFLAGS " ${thread_safety_flags_space_sep}")
endif()

# If we're using MSVC,
# always respect the optimization flags set by CMAKE_BUILD_TYPE instead.
if (NOT MSVC)

  # Build with optimization, unless we're in debug mode.
  if(COMPILER_RT_DEBUG)
    list(APPEND SANITIZER_COMMON_CFLAGS -O0)
  else()
    list(APPEND SANITIZER_COMMON_CFLAGS -O3)
  endif()
endif()

# Determine if we should restrict stack frame sizes.
# Stack frames on PowerPC, Mips, SystemZ and in debug build can be much larger than
# anticipated.
# FIXME: Fix all sanitizers and add -Wframe-larger-than to
# SANITIZER_COMMON_FLAGS
if(COMPILER_RT_HAS_WFRAME_LARGER_THAN_FLAG AND NOT COMPILER_RT_DEBUG
   AND NOT ${COMPILER_RT_DEFAULT_TARGET_ARCH} MATCHES "powerpc|mips|s390x")
  set(SANITIZER_LIMIT_FRAME_SIZE TRUE)
else()
  set(SANITIZER_LIMIT_FRAME_SIZE FALSE)
endif()

if(FUCHSIA OR UNIX)
  set(SANITIZER_USE_SYMBOLS TRUE)
else()
  set(SANITIZER_USE_SYMBOLS FALSE)
endif()

# Build sanitizer runtimes with debug info.
if(MSVC)
  # Use /Z7 instead of /Zi for the asan runtime. This avoids the LNK4099
  # warning from the MS linker complaining that it can't find the 'vc140.pdb'
  # file used by our object library compilations.
  list(APPEND SANITIZER_COMMON_CFLAGS /Z7)
  foreach(var_to_update
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "(^| )/Z[i7I]($| )" " /Z7 "
           "${var_to_update}" "${${var_to_update}}")
  endforeach()
elseif(COMPILER_RT_HAS_GLINE_TABLES_ONLY_FLAG AND NOT COMPILER_RT_DEBUG)
  list(APPEND SANITIZER_COMMON_CFLAGS -gline-tables-only)
elseif(COMPILER_RT_HAS_G_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -g)
endif()

if(LLVM_ENABLE_MODULES)
  # Sanitizers cannot be built with -fmodules. The interceptors intentionally
  # don't include system headers, which is incompatible with modules.
  list(APPEND SANITIZER_COMMON_CFLAGS -fno-modules)
endif()

# Turn off several warnings.
append_list_if(COMPILER_RT_HAS_WGNU_FLAG -Wno-gnu SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WVARIADIC_MACROS_FLAG -Wno-variadic-macros SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WC99_EXTENSIONS_FLAG -Wno-c99-extensions SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4146_FLAG /wd4146 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4291_FLAG /wd4291 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4391_FLAG /wd4391 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4722_FLAG /wd4722 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4800_FLAG /wd4800 SANITIZER_COMMON_CFLAGS)

append_list_if(MINGW -fms-extensions SANITIZER_COMMON_CFLAGS)

# Set common link flags.
append_list_if(COMPILER_RT_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs SANITIZER_COMMON_LINK_FLAGS)
append_list_if(COMPILER_RT_HAS_Z_TEXT -Wl,-z,text SANITIZER_COMMON_LINK_FLAGS)

if (COMPILER_RT_USE_BUILTINS_LIBRARY)
  string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
  if (ANDROID)
    append_list_if(COMPILER_RT_HAS_GCC_LIB gcc SANITIZER_COMMON_LINK_LIBS)
  else()
    append_list_if(COMPILER_RT_HAS_GCC_S_LIB gcc_s SANITIZER_COMMON_LINK_LIBS)
  endif()
endif()

append_list_if(COMPILER_RT_HAS_LIBC c SANITIZER_COMMON_LINK_LIBS)
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Fuchsia")
  list(APPEND SANITIZER_COMMON_LINK_LIBS zircon)
endif()

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Fuchsia")
  set(SANITIZER_NO_UNDEFINED_SYMBOLS_DEFAULT ON)
else()
  set(SANITIZER_NO_UNDEFINED_SYMBOLS_DEFAULT OFF)
endif()
option(SANITIZER_NO_UNDEFINED_SYMBOLS "Report error on unresolved symbol references" ${SANITIZER_NO_UNDEFINED_SYMBOLS_DEFAULT})
if (SANITIZER_NO_UNDEFINED_SYMBOLS)
  list(APPEND SANITIZER_COMMON_LINK_FLAGS -Wl,-z,defs)
endif()

# TODO: COMPILER_RT_COMMON_CFLAGS and COMPILER_RT_COMMON_LINK_FLAGS are
# intended for use in non-sanitizer runtimes such as libFuzzer, profile or XRay,
# move these higher to include common flags, then derive SANITIZER_COMMON_CFLAGS
# and SANITIZER_COMMON_LINK_FLAGS from those and append sanitizer-specific flags.
set(COMPILER_RT_COMMON_CFLAGS ${SANITIZER_COMMON_CFLAGS})
set(COMPILER_RT_COMMON_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

# We don't use the C++ standard library, so avoid including it by mistake.
append_list_if(COMPILER_RT_HAS_NOSTDINCXX_FLAG -nostdinc++ SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_NOSTDLIBXX_FLAG -nostdlib++ SANITIZER_COMMON_LINK_FLAGS)

# Remove -stdlib= which is unused when passing -nostdinc++...
string(REGEX MATCHALL "-stdlib=[a-zA-Z+]*" stdlib_flag "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-stdlib=[a-zA-Z+]*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# ...we need it to build some runtimes and tests so readd it where appropriate.
list(APPEND COMPILER_RT_COMMON_CFLAGS ${stdlib_flag})
list(APPEND COMPILER_RT_COMMON_LINK_FLAGS ${stdlib_flag})

macro(append_libcxx_libs var)
  if (${var}_INTREE)
    if (SANITIZER_USE_STATIC_LLVM_UNWINDER AND (TARGET unwind_static OR HAVE_LIBUNWIND))
      list(APPEND ${var}_LIBRARIES unwind_static)
    elseif (TARGET unwind_shared OR HAVE_LIBUNWIND)
      list(APPEND ${var}_LIBRARIES unwind_shared)
    endif()

    if (SANITIZER_USE_STATIC_CXX_ABI AND (TARGET cxxabi_static OR HAVE_LIBCXXABI))
      list(APPEND ${var}_LIBRARIES cxxabi_static)
    elseif (TARGET cxxabi_shared OR HAVE_LIBCXXABI)
      list(APPEND ${var}_LIBRARIES cxxabi_shared)
    endif()
  else()
    append_list_if(COMPILER_RT_HAS_LIBCXX c++ ${var}_LIBRARIES)
  endif()
endmacro()

if (SANITIZER_CXX_ABI_LIBNAME STREQUAL "libc++")
  append_libcxx_libs(SANITIZER_CXX_ABI)
elseif (SANITIZER_CXX_ABI_LIBNAME STREQUAL "libcxxabi")
  list(APPEND SANITIZER_CXX_ABI_LIBRARIES "c++abi")
elseif (SANITIZER_CXX_ABI_LIBNAME STREQUAL "libstdc++")
  append_list_if(COMPILER_RT_HAS_LIBSTDCXX stdc++ SANITIZER_CXX_ABI_LIBRARIES)
endif()

if (SANITIZER_TEST_CXX_LIBNAME STREQUAL "libc++")
  append_libcxx_libs(SANITIZER_TEST_CXX)
elseif (SANITIZER_TEST_CXX_LIBNAME STREQUAL "libstdc++")
  append_list_if(COMPILER_RT_HAS_LIBSTDCXX stdc++ SANITIZER_TEST_CXX_LIBRARIES)
endif()

# TODO: There's a lot of duplication across lib/*/tests/CMakeLists.txt files,
# move some of the common flags to COMPILER_RT_UNITTEST_CFLAGS.

# Unittests need access to C++ standard library.
string(APPEND COMPILER_RT_TEST_COMPILER_CFLAGS " ${stdlib_flag}")

# When cross-compiling, COMPILER_RT_TEST_COMPILER_CFLAGS help in compilation
# and linking of unittests.
string(REPLACE " " ";" COMPILER_RT_UNITTEST_CFLAGS "${COMPILER_RT_TEST_COMPILER_CFLAGS}")
set(COMPILER_RT_UNITTEST_LINK_FLAGS ${COMPILER_RT_UNITTEST_CFLAGS})

# Unittests support.
set(COMPILER_RT_GTEST_PATH ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest)
set(COMPILER_RT_GTEST_SOURCE ${COMPILER_RT_GTEST_PATH}/src/gtest-all.cc)
set(COMPILER_RT_GTEST_CFLAGS
  -DGTEST_NO_LLVM_SUPPORT=1
  -DGTEST_HAS_RTTI=0
  -I${COMPILER_RT_GTEST_PATH}/include
  -I${COMPILER_RT_GTEST_PATH}
)
if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  # FreeBSD has its pthread functions marked with thread safety annotations, but
  # googletest is not compatible with such annotations. Disable the thread
  # safety warnings-as-errors until googletest has been fixed.
  list(APPEND NO_THREAD_SAFETY_FLAGS ${THREAD_SAFETY_FLAGS})
  list(TRANSFORM NO_THREAD_SAFETY_FLAGS REPLACE "error=" "no-")
  list(APPEND COMPILER_RT_GTEST_CFLAGS ${NO_THREAD_SAFETY_FLAGS})
endif()

# Mocking support.
set(COMPILER_RT_GMOCK_PATH ${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock)
set(COMPILER_RT_GMOCK_SOURCE ${COMPILER_RT_GMOCK_PATH}/src/gmock-all.cc)
set(COMPILER_RT_GMOCK_CFLAGS
  -DGTEST_NO_LLVM_SUPPORT=1
  -DGTEST_HAS_RTTI=0
  -I${COMPILER_RT_GMOCK_PATH}/include
  -I${COMPILER_RT_GMOCK_PATH}
)

append_list_if(COMPILER_RT_DEBUG -DSANITIZER_DEBUG=1 COMPILER_RT_UNITTEST_CFLAGS)
append_list_if(COMPILER_RT_HAS_WCOVERED_SWITCH_DEFAULT_FLAG -Wno-covered-switch-default COMPILER_RT_UNITTEST_CFLAGS)
append_list_if(COMPILER_RT_HAS_WSUGGEST_OVERRIDE_FLAG -Wno-suggest-override COMPILER_RT_UNITTEST_CFLAGS)

if(MSVC)
  # gtest use a lot of stuff marked as deprecated on Windows.
  list(APPEND COMPILER_RT_GTEST_CFLAGS -Wno-deprecated-declarations)
endif()

# Warnings to turn off for all libraries, not just sanitizers.
append_string_if(COMPILER_RT_HAS_WUNUSED_PARAMETER_FLAG -Wno-unused-parameter CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

if (CMAKE_LINKER MATCHES "link.exe$")
  # Silence MSVC linker warnings caused by empty object files. The
  # sanitizer libraries intentionally use ifdefs that result in empty
  # files, rather than skipping these files in the build system.
  # Ideally, we would pass this flag only for the libraries that need
  # it, but CMake doesn't seem to have a way to set linker flags for
  # individual static libraries, so we enable the suppression flag for
  # the whole compiler-rt project.
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /IGNORE:4221")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
  set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> -X32_64 qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -X32_64 qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> -X32_64 q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> -X32_64 q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> -X32_64 <TARGET>")
  set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -X32_64 <TARGET>")
endif()

add_subdirectory(include)

option(COMPILER_RT_USE_LIBCXX
  "Enable compiler-rt to use libc++ from the source tree" ON)
if(COMPILER_RT_USE_LIBCXX)
  if(LLVM_ENABLE_PROJECTS_USED)
    # Don't use libcxx if LLVM_ENABLE_PROJECTS does not enable it.
    set(COMPILER_RT_LIBCXX_PATH ${LLVM_EXTERNAL_LIBCXX_SOURCE_DIR})
    set(COMPILER_RT_LIBCXXABI_PATH ${LLVM_EXTERNAL_LIBCXXABI_SOURCE_DIR})
  else()
    foreach(path IN ITEMS ${LLVM_MAIN_SRC_DIR}/projects/libcxx
                          ${LLVM_MAIN_SRC_DIR}/runtimes/libcxx
                          ${LLVM_MAIN_SRC_DIR}/../libcxx
                          ${LLVM_EXTERNAL_LIBCXX_SOURCE_DIR})
      if(IS_DIRECTORY ${path})
        set(COMPILER_RT_LIBCXX_PATH ${path})
        break()
      endif()
    endforeach()
    foreach(path IN ITEMS ${LLVM_MAIN_SRC_DIR}/projects/libcxxabi
                          ${LLVM_MAIN_SRC_DIR}/runtimes/libcxxabi
                          ${LLVM_MAIN_SRC_DIR}/../libcxxabi
                          ${LLVM_EXTERNAL_LIBCXXABI_SOURCE_DIR})
      if(IS_DIRECTORY ${path})
        set(COMPILER_RT_LIBCXXABI_PATH ${path})
        break()
      endif()
    endforeach()
  endif()
endif()

set(COMPILER_RT_LLD_PATH ${LLVM_MAIN_SRC_DIR}/tools/lld)
if(EXISTS ${COMPILER_RT_LLD_PATH}/ AND LLVM_TOOL_LLD_BUILD)
  set(COMPILER_RT_HAS_LLD TRUE)
else()
  set(COMPILER_RT_LLD_PATH ${LLVM_MAIN_SRC_DIR}/../lld)
  if(EXISTS ${COMPILER_RT_LLD_PATH}/ AND LLVM_TOOL_LLD_BUILD)
    set(COMPILER_RT_HAS_LLD TRUE)
  endif()
endif()

if(ANDROID)
  set(COMPILER_RT_HAS_LLD TRUE)
  set(COMPILER_RT_TEST_USE_LLD TRUE)
  append_list_if(COMPILER_RT_HAS_FUSE_LD_LLD_FLAG -fuse-ld=lld SANITIZER_COMMON_LINK_FLAGS)
  append_list_if(COMPILER_RT_HAS_LLD -fuse-ld=lld COMPILER_RT_UNITTEST_LINK_FLAGS)
  if(COMPILER_RT_HAS_FUSE_LD_LLD_FLAG)
    set(COMPILER_RT_HAS_GNU_VERSION_SCRIPT_COMPAT FALSE)
  endif()
endif()
pythonize_bool(COMPILER_RT_HAS_LLD)
pythonize_bool(COMPILER_RT_TEST_USE_LLD)

add_subdirectory(lib)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
  # Don't build llvm-lit for runtimes-build, it will clean up map_config.
  if (COMPILER_RT_STANDALONE_BUILD AND NOT LLVM_RUNTIMES_BUILD)
    # If we have a valid source tree, generate llvm-lit into the bin directory.
    # The user can still choose to have the check targets *use* a different lit
    # by specifying -DLLVM_EXTERNAL_LIT, but we generate it regardless.
    if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/llvm-lit)
      # Needed for lit support in standalone builds.
      include(AddLLVM)
      add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
    elseif(NOT EXISTS ${LLVM_EXTERNAL_LIT})
      message(WARNING "Could not find LLVM source directory and LLVM_EXTERNAL_LIT does not"
                       "point to a valid file.  You will not be able to run tests.")
    endif()
  endif()
endif()

add_subdirectory(tools)
