include(CMakePushCheckState)
include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include(CheckSymbolExists)
include(TestBigEndian)

function(check_linker_flag flag out_var)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${flag}")
  check_cxx_compiler_flag("" ${out_var})
  cmake_pop_check_state()
endfunction()

# CodeGen options.
check_cxx_compiler_flag(-fPIC                COMPILER_RT_HAS_FPIC_FLAG)
check_cxx_compiler_flag(-fPIE                COMPILER_RT_HAS_FPIE_FLAG)
check_cxx_compiler_flag(-fno-builtin         COMPILER_RT_HAS_FNO_BUILTIN_FLAG)
check_cxx_compiler_flag(-fno-exceptions      COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG)
check_cxx_compiler_flag(-fomit-frame-pointer COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG)
check_cxx_compiler_flag(-funwind-tables      COMPILER_RT_HAS_FUNWIND_TABLES_FLAG)
check_cxx_compiler_flag(-fno-stack-protector COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG)
check_cxx_compiler_flag(-fno-sanitize=safe-stack COMPILER_RT_HAS_FNO_SANITIZE_SAFE_STACK_FLAG)
check_cxx_compiler_flag(-fvisibility=hidden  COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG)
check_cxx_compiler_flag(-fno-rtti            COMPILER_RT_HAS_FNO_RTTI_FLAG)
check_cxx_compiler_flag(-ffreestanding       COMPILER_RT_HAS_FFREESTANDING_FLAG)
check_cxx_compiler_flag("-Werror -fno-function-sections" COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG)
check_cxx_compiler_flag(-std=c++11           COMPILER_RT_HAS_STD_CXX11_FLAG)
check_cxx_compiler_flag(-ftls-model=initial-exec COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC)
check_cxx_compiler_flag(-fno-lto             COMPILER_RT_HAS_FNO_LTO_FLAG)
check_cxx_compiler_flag(-msse3               COMPILER_RT_HAS_MSSE3_FLAG)

check_cxx_compiler_flag(/GR COMPILER_RT_HAS_GR_FLAG)
check_cxx_compiler_flag(/GS COMPILER_RT_HAS_GS_FLAG)
check_cxx_compiler_flag(/MT COMPILER_RT_HAS_MT_FLAG)
check_cxx_compiler_flag(/Oy COMPILER_RT_HAS_Oy_FLAG)

# Debug info flags.
check_cxx_compiler_flag(-gline-tables-only COMPILER_RT_HAS_GLINE_TABLES_ONLY_FLAG)
check_cxx_compiler_flag(-g COMPILER_RT_HAS_G_FLAG)
check_cxx_compiler_flag(/Zi COMPILER_RT_HAS_Zi_FLAG)

# Warnings.
check_cxx_compiler_flag(-Wall COMPILER_RT_HAS_WALL_FLAG)
check_cxx_compiler_flag(-Werror COMPILER_RT_HAS_WERROR_FLAG)
check_cxx_compiler_flag("-Werror -Wframe-larger-than=512" COMPILER_RT_HAS_WFRAME_LARGER_THAN_FLAG)
check_cxx_compiler_flag("-Werror -Wglobal-constructors"   COMPILER_RT_HAS_WGLOBAL_CONSTRUCTORS_FLAG)
check_cxx_compiler_flag("-Werror -Wc99-extensions"     COMPILER_RT_HAS_WC99_EXTENSIONS_FLAG)
check_cxx_compiler_flag("-Werror -Wgnu"                COMPILER_RT_HAS_WGNU_FLAG)
check_cxx_compiler_flag("-Werror -Wnon-virtual-dtor"   COMPILER_RT_HAS_WNON_VIRTUAL_DTOR_FLAG)
check_cxx_compiler_flag("-Werror -Wvariadic-macros"    COMPILER_RT_HAS_WVARIADIC_MACROS_FLAG)

check_cxx_compiler_flag(/W3 COMPILER_RT_HAS_W3_FLAG)
check_cxx_compiler_flag(/WX COMPILER_RT_HAS_WX_FLAG)
check_cxx_compiler_flag(/wd4146 COMPILER_RT_HAS_WD4146_FLAG)
check_cxx_compiler_flag(/wd4291 COMPILER_RT_HAS_WD4291_FLAG)
check_cxx_compiler_flag(/wd4391 COMPILER_RT_HAS_WD4391_FLAG)
check_cxx_compiler_flag(/wd4722 COMPILER_RT_HAS_WD4722_FLAG)
check_cxx_compiler_flag(/wd4800 COMPILER_RT_HAS_WD4800_FLAG)

# Symbols.
check_symbol_exists(__func__ "" COMPILER_RT_HAS_FUNC_SYMBOL)

# Libraries.
check_library_exists(c printf "" COMPILER_RT_HAS_LIBC)
check_library_exists(dl dlopen "" COMPILER_RT_HAS_LIBDL)
check_library_exists(rt shm_open "" COMPILER_RT_HAS_LIBRT)
check_library_exists(m pow "" COMPILER_RT_HAS_LIBM)
check_library_exists(pthread pthread_create "" COMPILER_RT_HAS_LIBPTHREAD)
check_library_exists(stdc++ __cxa_throw "" COMPILER_RT_HAS_LIBSTDCXX)

# Linker flags.
if(ANDROID)
  check_linker_flag("-Wl,-z,global" COMPILER_RT_HAS_Z_GLOBAL)
endif()

# Architectures.

# List of all architectures we can target.
set(COMPILER_RT_SUPPORTED_ARCH)

# Try to compile a very simple source file to ensure we can target the given
# platform. We use the results of these tests to build only the various target
# runtime libraries supported by our current compilers cross-compiling
# abilities.
set(SIMPLE_SOURCE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple.cc)
file(WRITE ${SIMPLE_SOURCE} "#include <stdlib.h>\n#include <limits>\nint main() {}\n")

function(check_compile_definition def argstring out_var)
  if("${def}" STREQUAL "")
    set(${out_var} TRUE PARENT_SCOPE)
    return()
  endif()
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${argstring}")
  check_symbol_exists(${def} "" ${out_var})
  cmake_pop_check_state()
endfunction()

# test_target_arch(<arch> <def> <target flags...>)
# Checks if architecture is supported: runs host compiler with provided
# flags to verify that:
#   1) <def> is defined (if non-empty)
#   2) simple file can be successfully built.
# If successful, saves target flags for this architecture.
macro(test_target_arch arch def)
  set(TARGET_${arch}_CFLAGS ${ARGN})
  set(argstring "")
  foreach(arg ${ARGN})
    set(argstring "${argstring} ${arg}")
  endforeach()
  check_compile_definition("${def}" "${argstring}" HAS_${arch}_DEF)
  if(NOT HAS_${arch}_DEF)
    set(CAN_TARGET_${arch} FALSE)
  else()
    set(argstring "${CMAKE_EXE_LINKER_FLAGS} ${argstring}")
    try_compile(CAN_TARGET_${arch} ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE}
                COMPILE_DEFINITIONS "${TARGET_${arch}_CFLAGS}"
                OUTPUT_VARIABLE TARGET_${arch}_OUTPUT
                CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${argstring}")
  endif()
  if(${CAN_TARGET_${arch}})
    list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
  elseif("${COMPILER_RT_TEST_TARGET_ARCH}" MATCHES "${arch}" AND
         COMPILER_RT_HAS_EXPLICIT_TEST_TARGET_TRIPLE)
    # Bail out if we cannot target the architecture we plan to test.
    message(FATAL_ERROR "Cannot compile for ${arch}:\n${TARGET_${arch}_OUTPUT}")
  endif()
endmacro()

# Add $arch as supported with no additional flags.
macro(add_default_target_arch arch)
  set(TARGET_${arch}_CFLAGS "")
  set(CAN_TARGET_${arch} 1)
  list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
endmacro()

macro(detect_target_arch)
  check_symbol_exists(__arm__ "" __ARM)
  check_symbol_exists(__aarch64__ "" __AARCH64)
  check_symbol_exists(__x86_64__ "" __X86_64)
  check_symbol_exists(__i686__ "" __I686)
  check_symbol_exists(__i386__ "" __I386)
  check_symbol_exists(__mips__ "" __MIPS)
  check_symbol_exists(__mips64__ "" __MIPS64)
  if(__ARM)
    add_default_target_arch(arm)
  elseif(__AARCH64)
    add_default_target_arch(aarch64)
  elseif(__X86_64)
    add_default_target_arch(x86_64)
  elseif(__I686)
    add_default_target_arch(i686)
  elseif(__I386)
    add_default_target_arch(i386)
  elseif(__MIPS64) # must be checked before __MIPS
    add_default_target_arch(mips64)
  elseif(__MIPS)
    add_default_target_arch(mips)
  endif()
endmacro()

# Detect whether the current target platform is 32-bit or 64-bit, and setup
# the correct commandline flags needed to attempt to target 32-bit and 64-bit.
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 4 AND
    NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Please use architecture with 4 or 8 byte pointers.")
endif()

# Generate the COMPILER_RT_SUPPORTED_ARCH list.
if(ANDROID)
  # Can't rely on LLVM_NATIVE_ARCH in cross-compilation.
  # Examine compiler output instead.
  detect_target_arch()
  set(COMPILER_RT_OS_SUFFIX "-android")
else()
  if("${LLVM_NATIVE_ARCH}" STREQUAL "X86")
    if(NOT MSVC)
      test_target_arch(x86_64 "" "-m64")
      # FIXME: We build runtimes for both i686 and i386, as "clang -m32" may
      # target different variant than "$CMAKE_C_COMPILER -m32". This part should
      # be gone after we resolve PR14109.
      test_target_arch(i686 __i686__ "-m32")
      test_target_arch(i386 __i386__ "-m32")
    else()
      if (CMAKE_SIZEOF_VOID_P EQUAL 4)
        test_target_arch(i386 "" "")
      else()
        test_target_arch(x86_64 "" "")
      endif()
    endif()
  elseif("${LLVM_NATIVE_ARCH}" STREQUAL "PowerPC")
    TEST_BIG_ENDIAN(HOST_IS_BIG_ENDIAN)
    if(HOST_IS_BIG_ENDIAN)
      test_target_arch(powerpc64 "" "-m64")
    else()
      test_target_arch(powerpc64le "" "-m64")
    endif()
  elseif("${LLVM_NATIVE_ARCH}" STREQUAL "Mips")
    # Gcc doesn't accept -m32/-m64 so we do the next best thing and use
    # -mips32r2/-mips64r2. We don't use -mips1/-mips3 because we want to match
    # clang's default CPU's. In the 64-bit case, we must also specify the ABI
    # since the default ABI differs between gcc and clang.
    # FIXME: Ideally, we would build the N32 library too.
    if("${COMPILER_RT_TEST_TARGET_ARCH}" MATCHES "mipsel|mips64el")
      # regex for mipsel, mips64el
      test_target_arch(mipsel "" "-mips32r2" "--target=mipsel-linux-gnu")
      test_target_arch(mips64el "" "-mips64r2" "-mabi=n64")
    else()
      test_target_arch(mips "" "-mips32r2" "--target=mips-linux-gnu")
      test_target_arch(mips64 "" "-mips64r2" "-mabi=n64")
    endif()
  elseif("${COMPILER_RT_TEST_TARGET_ARCH}" MATCHES "arm")
    test_target_arch(arm "" "-march=armv7-a")
  elseif("${COMPILER_RT_TEST_TARGET_ARCH}" MATCHES "aarch32")
    test_target_arch(aarch32 "" "-march=armv8-a")
  elseif("${COMPILER_RT_TEST_TARGET_ARCH}" MATCHES "aarch64")
    test_target_arch(aarch64 "" "-march=armv8-a")
  endif()
  set(COMPILER_RT_OS_SUFFIX "")
endif()

message(STATUS "Compiler-RT supported architectures: ${COMPILER_RT_SUPPORTED_ARCH}")

# Takes ${ARGN} and puts only supported architectures in @out_var list.
function(filter_available_targets out_var)
  set(archs)
  foreach(arch ${ARGN})
    list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
    if(NOT (ARCH_INDEX EQUAL -1) AND CAN_TARGET_${arch})
      list(APPEND archs ${arch})
    endif()
  endforeach()
  set(${out_var} ${archs} PARENT_SCOPE)
endfunction()

# Returns a list of architecture specific target cflags in @out_var list.
function(get_target_flags_for_arch arch out_var)
  list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
  if(ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "Unsupported architecture: ${arch}")
  else()
    set(${out_var} ${TARGET_${arch}_CFLAGS} PARENT_SCOPE)
  endif()
endfunction()

# Architectures supported by compiler-rt libraries.
filter_available_targets(SANITIZER_COMMON_SUPPORTED_ARCH
  x86_64 i386 i686 powerpc64 powerpc64le arm aarch64 mips mips64 mipsel mips64el)
# LSan and UBSan common files should be available on all architectures supported
# by other sanitizers (even if they build into dummy object files).
filter_available_targets(LSAN_COMMON_SUPPORTED_ARCH
  ${SANITIZER_COMMON_SUPPORTED_ARCH})
filter_available_targets(UBSAN_COMMON_SUPPORTED_ARCH
  ${SANITIZER_COMMON_SUPPORTED_ARCH})
filter_available_targets(ASAN_SUPPORTED_ARCH
  x86_64 i386 i686 powerpc64 powerpc64le arm mips mipsel mips64 mips64el)
filter_available_targets(DFSAN_SUPPORTED_ARCH x86_64 mips64 mips64el)
filter_available_targets(LSAN_SUPPORTED_ARCH x86_64 mips64 mips64el)
filter_available_targets(MSAN_SUPPORTED_ARCH x86_64 mips64 mips64el)
filter_available_targets(PROFILE_SUPPORTED_ARCH x86_64 i386 i686 arm mips mips64
  mipsel mips64el aarch64 powerpc64 powerpc64le)
filter_available_targets(TSAN_SUPPORTED_ARCH x86_64 mips64 mips64el)
filter_available_targets(UBSAN_SUPPORTED_ARCH x86_64 i386 i686 arm aarch64 mips
  mipsel mips64 mips64el powerpc64 powerpc64le)
filter_available_targets(SAFESTACK_SUPPORTED_ARCH x86_64 i386 i686)

if(ANDROID)
  set(OS_NAME "Android")
else()
  set(OS_NAME "${CMAKE_SYSTEM_NAME}")
endif()

if (SANITIZER_COMMON_SUPPORTED_ARCH AND NOT LLVM_USE_SANITIZER AND
    (OS_NAME MATCHES "Android|Darwin|Linux|FreeBSD" OR
    (OS_NAME MATCHES "Windows" AND MSVC)))
  set(COMPILER_RT_HAS_SANITIZER_COMMON TRUE)
else()
  set(COMPILER_RT_HAS_SANITIZER_COMMON FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND
    (NOT OS_NAME MATCHES "Windows" OR CMAKE_SIZEOF_VOID_P EQUAL 4))
  set(COMPILER_RT_HAS_INTERCEPTION TRUE)
else()
  set(COMPILER_RT_HAS_INTERCEPTION FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND ASAN_SUPPORTED_ARCH AND
    (NOT OS_NAME MATCHES "Windows" OR CMAKE_SIZEOF_VOID_P EQUAL 4))
  set(COMPILER_RT_HAS_ASAN TRUE)
else()
  set(COMPILER_RT_HAS_ASAN FALSE)
endif()

if (OS_NAME MATCHES "Linux|FreeBSD|Windows")
  set(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME TRUE)
else()
  set(COMPILER_RT_ASAN_HAS_STATIC_RUNTIME FALSE)
endif()

# TODO: Add builtins support.

if (COMPILER_RT_HAS_SANITIZER_COMMON AND DFSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux")
  set(COMPILER_RT_HAS_DFSAN TRUE)
else()
  set(COMPILER_RT_HAS_DFSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND LSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|FreeBSD")
  set(COMPILER_RT_HAS_LSAN TRUE)
else()
  set(COMPILER_RT_HAS_LSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND MSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux")
  set(COMPILER_RT_HAS_MSAN TRUE)
else()
  set(COMPILER_RT_HAS_MSAN FALSE)
endif()

if (PROFILE_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD")
  set(COMPILER_RT_HAS_PROFILE TRUE)
else()
  set(COMPILER_RT_HAS_PROFILE FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND TSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|FreeBSD")
  set(COMPILER_RT_HAS_TSAN TRUE)
else()
  set(COMPILER_RT_HAS_TSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND UBSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|Windows")
  set(COMPILER_RT_HAS_UBSAN TRUE)
else()
  set(COMPILER_RT_HAS_UBSAN FALSE)
endif()

# -msse3 flag is not valid for Mips therefore clang gives a warning
# message with -msse3. But check_c_compiler_flags() checks only for
# compiler error messages. Therefore COMPILER_RT_HAS_MSSE3_FLAG turns out to be
# true on Mips, so we make it false here.
if("${LLVM_NATIVE_ARCH}" STREQUAL "Mips")
  set(COMPILER_RT_HAS_MSSE3_FLAG FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND SAFESTACK_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD")
  set(COMPILER_RT_HAS_SAFESTACK TRUE)
else()
  set(COMPILER_RT_HAS_SAFESTACK FALSE)
endif()
