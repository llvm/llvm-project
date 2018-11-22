include(CMakePushCheckState)
include(CheckCCompilerFlag)
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

check_library_exists(c fopen "" COMPILER_RT_HAS_LIBC)
if (COMPILER_RT_USE_BUILTINS_LIBRARY)
  include(HandleCompilerRT)
  find_compiler_rt_library(builtins COMPILER_RT_BUILTINS_LIBRARY)
else()
  if (ANDROID)
    check_library_exists(gcc __gcc_personality_v0 "" COMPILER_RT_HAS_GCC_LIB)
  else()
    check_library_exists(gcc_s __gcc_personality_v0 "" COMPILER_RT_HAS_GCC_S_LIB)
  endif()
endif()

check_c_compiler_flag(-nodefaultlibs COMPILER_RT_HAS_NODEFAULTLIBS_FLAG)
if (COMPILER_RT_HAS_NODEFAULTLIBS_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nodefaultlibs")
  if (COMPILER_RT_HAS_LIBC)
    list(APPEND CMAKE_REQUIRED_LIBRARIES c)
  endif ()
  if (COMPILER_RT_USE_BUILTINS_LIBRARY)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${COMPILER_RT_BUILTINS_LIBRARY}")
  elseif (COMPILER_RT_HAS_GCC_S_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES gcc_s)
  elseif (COMPILER_RT_HAS_GCC_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES gcc)
  endif ()
  if (MINGW)
    # Mingw64 requires quite a few "C" runtime libraries in order for basic
    # programs to link successfully with -nodefaultlibs.
    if (COMPILER_RT_USE_BUILTINS_LIBRARY)
      set(MINGW_RUNTIME ${COMPILER_RT_BUILTINS_LIBRARY})
    else ()
      set(MINGW_RUNTIME gcc_s gcc)
    endif()
    set(MINGW_LIBRARIES mingw32 ${MINGW_RUNTIME} moldname mingwex msvcrt advapi32
                        shell32 user32 kernel32 mingw32 ${MINGW_RUNTIME}
                        moldname mingwex msvcrt)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${MINGW_LIBRARIES})
  endif()
endif ()

# CodeGen options.
check_c_compiler_flag(-ffreestanding         COMPILER_RT_HAS_FFREESTANDING_FLAG)
check_cxx_compiler_flag(-fPIC                COMPILER_RT_HAS_FPIC_FLAG)
check_cxx_compiler_flag(-fPIE                COMPILER_RT_HAS_FPIE_FLAG)
check_cxx_compiler_flag(-fno-builtin         COMPILER_RT_HAS_FNO_BUILTIN_FLAG)
check_cxx_compiler_flag(-fno-exceptions      COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG)
check_cxx_compiler_flag(-fomit-frame-pointer COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG)
check_cxx_compiler_flag(-funwind-tables      COMPILER_RT_HAS_FUNWIND_TABLES_FLAG)
check_cxx_compiler_flag(-fno-stack-protector COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG)
check_cxx_compiler_flag(-fno-sanitize=safe-stack COMPILER_RT_HAS_FNO_SANITIZE_SAFE_STACK_FLAG)
check_cxx_compiler_flag(-fvisibility=hidden  COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG)
check_cxx_compiler_flag(-frtti               COMPILER_RT_HAS_FRTTI_FLAG)
check_cxx_compiler_flag(-fno-rtti            COMPILER_RT_HAS_FNO_RTTI_FLAG)
check_cxx_compiler_flag("-Werror -fno-function-sections" COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG)
check_cxx_compiler_flag(-std=c++11           COMPILER_RT_HAS_STD_CXX11_FLAG)
check_cxx_compiler_flag(-ftls-model=initial-exec COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC)
check_cxx_compiler_flag(-fno-lto             COMPILER_RT_HAS_FNO_LTO_FLAG)
check_cxx_compiler_flag("-Werror -msse3" COMPILER_RT_HAS_MSSE3_FLAG)
check_cxx_compiler_flag("-Werror -msse4.2"   COMPILER_RT_HAS_MSSE4_2_FLAG)
check_cxx_compiler_flag(--sysroot=.          COMPILER_RT_HAS_SYSROOT_FLAG)
check_cxx_compiler_flag("-Werror -mcrc"      COMPILER_RT_HAS_MCRC_FLAG)

if(NOT WIN32 AND NOT CYGWIN)
  # MinGW warns if -fvisibility-inlines-hidden is used.
  check_cxx_compiler_flag("-fvisibility-inlines-hidden" COMPILER_RT_HAS_FVISIBILITY_INLINES_HIDDEN_FLAG)
endif()

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
check_cxx_compiler_flag("-Werror -Wunused-parameter"   COMPILER_RT_HAS_WUNUSED_PARAMETER_FLAG)
check_cxx_compiler_flag("-Werror -Wcovered-switch-default" COMPILER_RT_HAS_WCOVERED_SWITCH_DEFAULT_FLAG)

check_cxx_compiler_flag(/W4 COMPILER_RT_HAS_W4_FLAG)
check_cxx_compiler_flag(/WX COMPILER_RT_HAS_WX_FLAG)
check_cxx_compiler_flag(/wd4146 COMPILER_RT_HAS_WD4146_FLAG)
check_cxx_compiler_flag(/wd4291 COMPILER_RT_HAS_WD4291_FLAG)
check_cxx_compiler_flag(/wd4221 COMPILER_RT_HAS_WD4221_FLAG)
check_cxx_compiler_flag(/wd4391 COMPILER_RT_HAS_WD4391_FLAG)
check_cxx_compiler_flag(/wd4722 COMPILER_RT_HAS_WD4722_FLAG)
check_cxx_compiler_flag(/wd4800 COMPILER_RT_HAS_WD4800_FLAG)

# Symbols.
check_symbol_exists(__func__ "" COMPILER_RT_HAS_FUNC_SYMBOL)

# Libraries.
check_library_exists(dl dlopen "" COMPILER_RT_HAS_LIBDL)
check_library_exists(rt shm_open "" COMPILER_RT_HAS_LIBRT)
check_library_exists(m pow "" COMPILER_RT_HAS_LIBM)
check_library_exists(pthread pthread_create "" COMPILER_RT_HAS_LIBPTHREAD)

# Look for terminfo library, used in unittests that depend on LLVMSupport.
if(LLVM_ENABLE_TERMINFO)
  foreach(library terminfo tinfo curses ncurses ncursesw)
    string(TOUPPER ${library} library_suffix)
    check_library_exists(
      ${library} setupterm "" COMPILER_RT_HAS_TERMINFO_${library_suffix})
    if(COMPILER_RT_HAS_TERMINFO_${library_suffix})
      set(COMPILER_RT_HAS_TERMINFO TRUE)
      set(COMPILER_RT_TERMINFO_LIB "${library}")
      break()
    endif()
  endforeach()
endif()

if (ANDROID AND COMPILER_RT_HAS_LIBDL)
  # Android's libstdc++ has a dependency on libdl.
  list(APPEND CMAKE_REQUIRED_LIBRARIES dl)
endif()
check_library_exists(c++ __cxa_throw "" COMPILER_RT_HAS_LIBCXX)
check_library_exists(stdc++ __cxa_throw "" COMPILER_RT_HAS_LIBSTDCXX)

# Linker flags.
if(ANDROID)
  check_linker_flag("-Wl,-z,global" COMPILER_RT_HAS_Z_GLOBAL)
  check_library_exists(log __android_log_write "" COMPILER_RT_HAS_LIBLOG)
endif()

# Architectures.

# List of all architectures we can target.
set(COMPILER_RT_SUPPORTED_ARCH)

# Try to compile a very simple source file to ensure we can target the given
# platform. We use the results of these tests to build only the various target
# runtime libraries supported by our current compilers cross-compiling
# abilities.
set(SIMPLE_SOURCE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple.cc)
file(WRITE ${SIMPLE_SOURCE} "#include <stdlib.h>\n#include <stdio.h>\nint main() { printf(\"hello, world\"); }\n")

# Detect whether the current target platform is 32-bit or 64-bit, and setup
# the correct commandline flags needed to attempt to target 32-bit and 64-bit.
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 4 AND
    NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Please use architecture with 4 or 8 byte pointers.")
endif()

test_targets()

# Returns a list of architecture specific target cflags in @out_var list.
function(get_target_flags_for_arch arch out_var)
  list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
  if(ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "Unsupported architecture: ${arch}")
  else()
    if (NOT APPLE)
      set(${out_var} ${TARGET_${arch}_CFLAGS} PARENT_SCOPE)
    else()
      # This is only called in constructing cflags for tests executing on the
      # host. This will need to all be cleaned up to support building tests
      # for cross-targeted hardware (i.e. iOS).
      set(${out_var} -arch ${arch} PARENT_SCOPE)
    endif()
  endif()
endfunction()

# Returns a compiler and CFLAGS that should be used to run tests for the
# specific architecture.  When cross-compiling, this is controled via
# COMPILER_RT_TEST_COMPILER and COMPILER_RT_TEST_COMPILER_CFLAGS.
macro(get_test_cc_for_arch arch cc_out cflags_out)
  if(ANDROID OR ${arch} MATCHES "arm|aarch64")
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(${cc_out} ${COMPILER_RT_TEST_COMPILER})
    set(${cflags_out} ${COMPILER_RT_TEST_COMPILER_CFLAGS})
  else()
    get_target_flags_for_arch(${arch} ${cflags_out})
    if(APPLE)
      list(APPEND ${cflags_out} ${DARWIN_osx_CFLAGS})
    endif()
    string(REPLACE ";" " " ${cflags_out} "${${cflags_out}}")
  endif()
endmacro()

set(ARM64 aarch64)
set(ARM32 arm armhf)
set(HEXAGON hexagon)
set(X86 i386)
set(X86_64 x86_64)
set(MIPS32 mips mipsel)
set(MIPS64 mips64 mips64el)
set(PPC64 powerpc64 powerpc64le)
set(RISCV32 riscv32)
set(RISCV64 riscv64)
set(S390X s390x)
set(WASM32 wasm32)
set(WASM64 wasm64)

if(APPLE)
  set(ARM64 arm64)
  set(ARM32 armv7 armv7s armv7k)
  set(X86_64 x86_64 x86_64h)
endif()

set(ALL_SANITIZER_COMMON_SUPPORTED_ARCH ${X86} ${X86_64} ${PPC64}
    ${ARM32} ${ARM64} ${MIPS32} ${MIPS64} ${S390X})
set(ALL_ASAN_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM32} ${ARM64}
    ${MIPS32} ${MIPS64} ${PPC64} ${S390X})
set(ALL_DFSAN_SUPPORTED_ARCH ${X86_64} ${MIPS64} ${ARM64})
set(ALL_FUZZER_SUPPORTED_ARCH ${X86_64} ${ARM64})

if(APPLE)
  set(ALL_LSAN_SUPPORTED_ARCH ${X86} ${X86_64} ${MIPS64} ${ARM64})
else()
  set(ALL_LSAN_SUPPORTED_ARCH ${X86} ${X86_64} ${MIPS64} ${ARM64} ${ARM32} ${PPC64})
endif()
set(ALL_MSAN_SUPPORTED_ARCH ${X86_64} ${MIPS64} ${ARM64} ${PPC64})
set(ALL_HWASAN_SUPPORTED_ARCH ${X86_64} ${ARM64})
set(ALL_PROFILE_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM32} ${ARM64} ${PPC64}
    ${MIPS32} ${MIPS64} ${S390X})
set(ALL_TSAN_SUPPORTED_ARCH ${X86_64} ${MIPS64} ${ARM64} ${PPC64})
set(ALL_UBSAN_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM32} ${ARM64}
    ${MIPS32} ${MIPS64} ${PPC64} ${S390X})
set(ALL_SAFESTACK_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM64} ${MIPS32} ${MIPS64})
set(ALL_CFI_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM32} ${ARM64} ${MIPS64})
set(ALL_ESAN_SUPPORTED_ARCH ${X86_64} ${MIPS64})
set(ALL_SCUDO_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM32} ${ARM64} ${MIPS32} ${MIPS64} ${PPC64})
if(APPLE)
set(ALL_XRAY_SUPPORTED_ARCH ${X86_64})
else()
set(ALL_XRAY_SUPPORTED_ARCH ${X86_64} ${ARM32} ${ARM64} ${MIPS32} ${MIPS64} powerpc64le)
endif()
set(ALL_SHADOWCALLSTACK_SUPPORTED_ARCH ${X86_64} ${ARM64})

if(APPLE)
  include(CompilerRTDarwinUtils)

  find_darwin_sdk_dir(DARWIN_osx_SYSROOT macosx)
  find_darwin_sdk_dir(DARWIN_iossim_SYSROOT iphonesimulator)
  find_darwin_sdk_dir(DARWIN_ios_SYSROOT iphoneos)
  find_darwin_sdk_dir(DARWIN_watchossim_SYSROOT watchsimulator)
  find_darwin_sdk_dir(DARWIN_watchos_SYSROOT watchos)
  find_darwin_sdk_dir(DARWIN_tvossim_SYSROOT appletvsimulator)
  find_darwin_sdk_dir(DARWIN_tvos_SYSROOT appletvos)

  if(NOT DARWIN_osx_SYSROOT)
    if(EXISTS /usr/include)
      set(DARWIN_osx_SYSROOT /)
    else()
      message(ERROR "Could not detect OS X Sysroot. Either install Xcode or the Apple Command Line Tools")
    endif()
  endif()

  if(COMPILER_RT_ENABLE_IOS)
    list(APPEND DARWIN_EMBEDDED_PLATFORMS ios)
    set(DARWIN_ios_MIN_VER_FLAG -miphoneos-version-min)
    set(DARWIN_ios_SANITIZER_MIN_VER_FLAG
      ${DARWIN_ios_MIN_VER_FLAG}=8.0)
  endif()
  if(COMPILER_RT_ENABLE_WATCHOS)
    list(APPEND DARWIN_EMBEDDED_PLATFORMS watchos)
    set(DARWIN_watchos_MIN_VER_FLAG -mwatchos-version-min)
    set(DARWIN_watchos_SANITIZER_MIN_VER_FLAG
      ${DARWIN_watchos_MIN_VER_FLAG}=2.0)
  endif()
  if(COMPILER_RT_ENABLE_TVOS)
    list(APPEND DARWIN_EMBEDDED_PLATFORMS tvos)
    set(DARWIN_tvos_MIN_VER_FLAG -mtvos-version-min)
    set(DARWIN_tvos_SANITIZER_MIN_VER_FLAG
      ${DARWIN_tvos_MIN_VER_FLAG}=9.0)
  endif()

  # Note: In order to target x86_64h on OS X the minimum deployment target must
  # be 10.8 or higher.
  set(SANITIZER_COMMON_SUPPORTED_OS osx)
  set(PROFILE_SUPPORTED_OS osx)
  set(TSAN_SUPPORTED_OS osx)
  set(XRAY_SUPPORTED_OS osx)
  if(NOT SANITIZER_MIN_OSX_VERSION)
    string(REGEX MATCH "-mmacosx-version-min=([.0-9]+)"
           MACOSX_VERSION_MIN_FLAG "${CMAKE_CXX_FLAGS}")
    if(MACOSX_VERSION_MIN_FLAG)
      set(SANITIZER_MIN_OSX_VERSION "${CMAKE_MATCH_1}")
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET)
      set(SANITIZER_MIN_OSX_VERSION ${CMAKE_OSX_DEPLOYMENT_TARGET})
    else()
      set(SANITIZER_MIN_OSX_VERSION 10.9)
    endif()
    if(SANITIZER_MIN_OSX_VERSION VERSION_LESS "10.7")
      message(FATAL_ERROR "macOS deployment target '${SANITIZER_MIN_OSX_VERSION}' is too old.")
    endif()
    if(SANITIZER_MIN_OSX_VERSION VERSION_GREATER "10.9")
      message(WARNING "macOS deployment target '${SANITIZER_MIN_OSX_VERSION}' is too new, setting to '10.9' instead.")
      set(SANITIZER_MIN_OSX_VERSION 10.9)
    endif()
  endif()

  # We're setting the flag manually for each target OS
  set(CMAKE_OSX_DEPLOYMENT_TARGET "")
  
  set(DARWIN_COMMON_CFLAGS -stdlib=libc++)
  set(DARWIN_COMMON_LINK_FLAGS
    -stdlib=libc++
    -lc++
    -lc++abi)
  
  check_linker_flag("-fapplication-extension" COMPILER_RT_HAS_APP_EXTENSION)
  if(COMPILER_RT_HAS_APP_EXTENSION)
    list(APPEND DARWIN_COMMON_LINK_FLAGS "-fapplication-extension")
  endif()

  set(DARWIN_osx_CFLAGS
    ${DARWIN_COMMON_CFLAGS}
    -mmacosx-version-min=${SANITIZER_MIN_OSX_VERSION})
  set(DARWIN_osx_LINK_FLAGS
    ${DARWIN_COMMON_LINK_FLAGS}
    -mmacosx-version-min=${SANITIZER_MIN_OSX_VERSION})

  if(DARWIN_osx_SYSROOT)
    list(APPEND DARWIN_osx_CFLAGS -isysroot ${DARWIN_osx_SYSROOT})
    list(APPEND DARWIN_osx_LINK_FLAGS -isysroot ${DARWIN_osx_SYSROOT})
  endif()

  # Figure out which arches to use for each OS
  darwin_get_toolchain_supported_archs(toolchain_arches)
  message(STATUS "Toolchain supported arches: ${toolchain_arches}")
  
  if(NOT MACOSX_VERSION_MIN_FLAG)
    darwin_test_archs(osx
      DARWIN_osx_ARCHS
      ${toolchain_arches})
    message(STATUS "OSX supported arches: ${DARWIN_osx_ARCHS}")
    foreach(arch ${DARWIN_osx_ARCHS})
      list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
      set(CAN_TARGET_${arch} 1)
    endforeach()

    foreach(platform ${DARWIN_EMBEDDED_PLATFORMS})
      if(DARWIN_${platform}sim_SYSROOT)
        set(DARWIN_${platform}sim_CFLAGS
          ${DARWIN_COMMON_CFLAGS}
          ${DARWIN_${platform}_SANITIZER_MIN_VER_FLAG}
          -isysroot ${DARWIN_${platform}sim_SYSROOT})
        set(DARWIN_${platform}sim_LINK_FLAGS
          ${DARWIN_COMMON_LINK_FLAGS}
          ${DARWIN_${platform}_SANITIZER_MIN_VER_FLAG}
          -isysroot ${DARWIN_${platform}sim_SYSROOT})

        set(DARWIN_${platform}sim_SKIP_CC_KEXT On)
        darwin_test_archs(${platform}sim
          DARWIN_${platform}sim_ARCHS
          ${toolchain_arches})
        message(STATUS "${platform} Simulator supported arches: ${DARWIN_${platform}sim_ARCHS}")
        if(DARWIN_${platform}sim_ARCHS)
          list(APPEND SANITIZER_COMMON_SUPPORTED_OS ${platform}sim)
          list(APPEND PROFILE_SUPPORTED_OS ${platform}sim)
          list(APPEND TSAN_SUPPORTED_OS ${platform}sim)
        endif()
        foreach(arch ${DARWIN_${platform}sim_ARCHS})
          list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
          set(CAN_TARGET_${arch} 1)
        endforeach()
      endif()

      if(DARWIN_${platform}_SYSROOT)
        set(DARWIN_${platform}_CFLAGS
          ${DARWIN_COMMON_CFLAGS}
          ${DARWIN_${platform}_SANITIZER_MIN_VER_FLAG}
          -isysroot ${DARWIN_${platform}_SYSROOT})
        set(DARWIN_${platform}_LINK_FLAGS
          ${DARWIN_COMMON_LINK_FLAGS}
          ${DARWIN_${platform}_SANITIZER_MIN_VER_FLAG}
          -isysroot ${DARWIN_${platform}_SYSROOT})

        darwin_test_archs(${platform}
          DARWIN_${platform}_ARCHS
          ${toolchain_arches})
        message(STATUS "${platform} supported arches: ${DARWIN_${platform}_ARCHS}")
        if(DARWIN_${platform}_ARCHS)
          list(APPEND SANITIZER_COMMON_SUPPORTED_OS ${platform})
          list(APPEND PROFILE_SUPPORTED_OS ${platform})

          list_intersect(DARWIN_${platform}_TSAN_ARCHS DARWIN_${platform}_ARCHS ALL_TSAN_SUPPORTED_ARCH)
          if(DARWIN_${platform}_TSAN_ARCHS)
            list(APPEND TSAN_SUPPORTED_OS ${platform})
          endif()
        endif()
        foreach(arch ${DARWIN_${platform}_ARCHS})
          list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
          set(CAN_TARGET_${arch} 1)
        endforeach()
      endif()
    endforeach()
  endif()

  # for list_intersect
  include(CompilerRTUtils)

  list_intersect(SANITIZER_COMMON_SUPPORTED_ARCH
    ALL_SANITIZER_COMMON_SUPPORTED_ARCH
    COMPILER_RT_SUPPORTED_ARCH
    )
  set(LSAN_COMMON_SUPPORTED_ARCH ${SANITIZER_COMMON_SUPPORTED_ARCH})
  set(UBSAN_COMMON_SUPPORTED_ARCH ${SANITIZER_COMMON_SUPPORTED_ARCH})
  list_intersect(ASAN_SUPPORTED_ARCH
    ALL_ASAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(DFSAN_SUPPORTED_ARCH
    ALL_DFSAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(LSAN_SUPPORTED_ARCH
    ALL_LSAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(MSAN_SUPPORTED_ARCH
    ALL_MSAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(HWASAN_SUPPORTED_ARCH
    ALL_HWASAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(PROFILE_SUPPORTED_ARCH
    ALL_PROFILE_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(TSAN_SUPPORTED_ARCH
    ALL_TSAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(UBSAN_SUPPORTED_ARCH
    ALL_UBSAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(SAFESTACK_SUPPORTED_ARCH
    ALL_SAFESTACK_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(CFI_SUPPORTED_ARCH
    ALL_CFI_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(ESAN_SUPPORTED_ARCH
    ALL_ESAN_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(SCUDO_SUPPORTED_ARCH
    ALL_SCUDO_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(FUZZER_SUPPORTED_ARCH
    ALL_FUZZER_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(XRAY_SUPPORTED_ARCH
    ALL_XRAY_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)
  list_intersect(SHADOWCALLSTACK_SUPPORTED_ARCH
    ALL_SHADOWCALLSTACK_SUPPORTED_ARCH
    SANITIZER_COMMON_SUPPORTED_ARCH)

else()
  # Architectures supported by compiler-rt libraries.
  filter_available_targets(SANITIZER_COMMON_SUPPORTED_ARCH
    ${ALL_SANITIZER_COMMON_SUPPORTED_ARCH})
  # LSan and UBSan common files should be available on all architectures
  # supported by other sanitizers (even if they build into dummy object files).
  filter_available_targets(LSAN_COMMON_SUPPORTED_ARCH
    ${SANITIZER_COMMON_SUPPORTED_ARCH})
  filter_available_targets(UBSAN_COMMON_SUPPORTED_ARCH
    ${SANITIZER_COMMON_SUPPORTED_ARCH})
  filter_available_targets(ASAN_SUPPORTED_ARCH ${ALL_ASAN_SUPPORTED_ARCH})
  filter_available_targets(FUZZER_SUPPORTED_ARCH ${ALL_FUZZER_SUPPORTED_ARCH})
  filter_available_targets(DFSAN_SUPPORTED_ARCH ${ALL_DFSAN_SUPPORTED_ARCH})
  filter_available_targets(LSAN_SUPPORTED_ARCH ${ALL_LSAN_SUPPORTED_ARCH})
  filter_available_targets(MSAN_SUPPORTED_ARCH ${ALL_MSAN_SUPPORTED_ARCH})
  filter_available_targets(HWASAN_SUPPORTED_ARCH ${ALL_HWASAN_SUPPORTED_ARCH})
  filter_available_targets(PROFILE_SUPPORTED_ARCH ${ALL_PROFILE_SUPPORTED_ARCH})
  filter_available_targets(TSAN_SUPPORTED_ARCH ${ALL_TSAN_SUPPORTED_ARCH})
  filter_available_targets(UBSAN_SUPPORTED_ARCH ${ALL_UBSAN_SUPPORTED_ARCH})
  filter_available_targets(SAFESTACK_SUPPORTED_ARCH
    ${ALL_SAFESTACK_SUPPORTED_ARCH})
  filter_available_targets(CFI_SUPPORTED_ARCH ${ALL_CFI_SUPPORTED_ARCH})
  filter_available_targets(ESAN_SUPPORTED_ARCH ${ALL_ESAN_SUPPORTED_ARCH})
  filter_available_targets(SCUDO_SUPPORTED_ARCH ${ALL_SCUDO_SUPPORTED_ARCH})
  filter_available_targets(XRAY_SUPPORTED_ARCH ${ALL_XRAY_SUPPORTED_ARCH})
  filter_available_targets(SHADOWCALLSTACK_SUPPORTED_ARCH
    ${ALL_SHADOWCALLSTACK_SUPPORTED_ARCH})
endif()

if (MSVC)
  # See if the DIA SDK is available and usable.
  set(MSVC_DIA_SDK_DIR "$ENV{VSINSTALLDIR}DIA SDK")
  if (IS_DIRECTORY ${MSVC_DIA_SDK_DIR})
    set(CAN_SYMBOLIZE 1)
  else()
    set(CAN_SYMBOLIZE 0)
  endif()
else()
  set(CAN_SYMBOLIZE 1)
endif()

find_program(GOLD_EXECUTABLE NAMES ${LLVM_DEFAULT_TARGET_TRIPLE}-ld.gold ld.gold ${LLVM_DEFAULT_TARGET_TRIPLE}-ld ld DOC "The gold linker")

if(COMPILER_RT_SUPPORTED_ARCH)
  list(REMOVE_DUPLICATES COMPILER_RT_SUPPORTED_ARCH)
endif()
message(STATUS "Compiler-RT supported architectures: ${COMPILER_RT_SUPPORTED_ARCH}")

if(ANDROID)
  set(OS_NAME "Android")
else()
  set(OS_NAME "${CMAKE_SYSTEM_NAME}")
endif()

set(ALL_SANITIZERS asan;dfsan;msan;hwasan;tsan;safestack;cfi;esan;scudo;ubsan_minimal)
set(COMPILER_RT_SANITIZERS_TO_BUILD all CACHE STRING
    "sanitizers to build if supported on the target (all;${ALL_SANITIZERS})")
list_replace(COMPILER_RT_SANITIZERS_TO_BUILD all "${ALL_SANITIZERS}")

if (SANITIZER_COMMON_SUPPORTED_ARCH AND NOT LLVM_USE_SANITIZER AND
    (OS_NAME MATCHES "Android|Darwin|Linux|FreeBSD|NetBSD|OpenBSD|Fuchsia|SunOS" OR
    (OS_NAME MATCHES "Windows" AND NOT CYGWIN AND
        (NOT MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))))
  set(COMPILER_RT_HAS_SANITIZER_COMMON TRUE)
else()
  set(COMPILER_RT_HAS_SANITIZER_COMMON FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON)
  set(COMPILER_RT_HAS_INTERCEPTION TRUE)
else()
  set(COMPILER_RT_HAS_INTERCEPTION FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND ASAN_SUPPORTED_ARCH AND
    NOT OS_NAME MATCHES "OpenBSD")
  set(COMPILER_RT_HAS_ASAN TRUE)
else()
  set(COMPILER_RT_HAS_ASAN FALSE)
endif()

if (OS_NAME MATCHES "Linux|FreeBSD|Windows|NetBSD|SunOS")
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
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|NetBSD")
  set(COMPILER_RT_HAS_LSAN TRUE)
else()
  set(COMPILER_RT_HAS_LSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND MSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|FreeBSD|NetBSD")
  set(COMPILER_RT_HAS_MSAN TRUE)
else()
  set(COMPILER_RT_HAS_MSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND HWASAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|Android")
  set(COMPILER_RT_HAS_HWASAN TRUE)
else()
  set(COMPILER_RT_HAS_HWASAN FALSE)
endif()

if (PROFILE_SUPPORTED_ARCH AND NOT LLVM_USE_SANITIZER AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|Windows|Android|Fuchsia|SunOS")
  set(COMPILER_RT_HAS_PROFILE TRUE)
else()
  set(COMPILER_RT_HAS_PROFILE FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND TSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|Android|NetBSD")
  set(COMPILER_RT_HAS_TSAN TRUE)
else()
  set(COMPILER_RT_HAS_TSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND UBSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|NetBSD|OpenBSD|Windows|Android|Fuchsia|SunOS")
  set(COMPILER_RT_HAS_UBSAN TRUE)
else()
  set(COMPILER_RT_HAS_UBSAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND UBSAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|FreeBSD|NetBSD|OpenBSD|Android|Darwin")
  set(COMPILER_RT_HAS_UBSAN_MINIMAL TRUE)
else()
  set(COMPILER_RT_HAS_UBSAN_MINIMAL FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND SAFESTACK_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|NetBSD")
  set(COMPILER_RT_HAS_SAFESTACK TRUE)
else()
  set(COMPILER_RT_HAS_SAFESTACK FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND CFI_SUPPORTED_ARCH)
  set(COMPILER_RT_HAS_CFI TRUE)
else()
  set(COMPILER_RT_HAS_CFI FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND ESAN_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|FreeBSD")
  set(COMPILER_RT_HAS_ESAN TRUE)
else()
  set(COMPILER_RT_HAS_ESAN FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND SCUDO_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|Android|Fuchsia")
  set(COMPILER_RT_HAS_SCUDO TRUE)
else()
  set(COMPILER_RT_HAS_SCUDO FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND XRAY_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Darwin|Linux|FreeBSD|NetBSD|OpenBSD|Fuchsia")
  set(COMPILER_RT_HAS_XRAY TRUE)
else()
  set(COMPILER_RT_HAS_XRAY FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND FUZZER_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Android|Darwin|Linux|NetBSD|FreeBSD|OpenBSD|Fuchsia|Windows" AND
    # TODO: Support builds with MSVC.
    NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND
    NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
  set(COMPILER_RT_HAS_FUZZER TRUE)
else()
  set(COMPILER_RT_HAS_FUZZER FALSE)
endif()

if (COMPILER_RT_HAS_SANITIZER_COMMON AND SHADOWCALLSTACK_SUPPORTED_ARCH AND
    OS_NAME MATCHES "Linux|Android")
  set(COMPILER_RT_HAS_SHADOWCALLSTACK TRUE)
else()
  set(COMPILER_RT_HAS_SHADOWCALLSTACK FALSE)
endif()
