include(BuiltinTests)

# Make all the tests only check the compiler
set(TEST_COMPILE_ONLY On)

builtin_check_c_compiler_flag(-fPIC                 COMPILER_RT_HAS_FPIC_FLAG)
builtin_check_c_compiler_flag(-fPIE                 COMPILER_RT_HAS_FPIE_FLAG)
builtin_check_c_compiler_flag(-fno-builtin          COMPILER_RT_HAS_FNO_BUILTIN_FLAG)
builtin_check_c_compiler_flag(-std=c99              COMPILER_RT_HAS_STD_C99_FLAG)
builtin_check_c_compiler_flag(-fvisibility=hidden   COMPILER_RT_HAS_VISIBILITY_HIDDEN_FLAG)
builtin_check_c_compiler_flag(-fomit-frame-pointer  COMPILER_RT_HAS_OMIT_FRAME_POINTER_FLAG)
builtin_check_c_compiler_flag(-ffreestanding        COMPILER_RT_HAS_FREESTANDING_FLAG)
builtin_check_c_compiler_flag(-mfloat-abi=soft      COMPILER_RT_HAS_FLOAT_ABI_SOFT_FLAG)
builtin_check_c_compiler_flag(-mfloat-abi=hard      COMPILER_RT_HAS_FLOAT_ABI_HARD_FLAG)
builtin_check_c_compiler_flag(-static               COMPILER_RT_HAS_STATIC_FLAG)

set(ARM64 aarch64)
set(ARM32 arm armhf)
set(X86 i386 i686)
set(X86_64 x86_64)
set(MIPS32 mips mipsel)
set(MIPS64 mips64 mips64el)
set(PPC64 powerpc64 powerpc64le)
set(WASM32 wasm32)
set(WASM64 wasm64)

if(APPLE)
  set(ARM64 arm64)
  set(ARM32 armv7 armv7k armv7s)
  set(X86_64 x86_64 x86_64h)
endif()

set(ALL_BUILTIN_SUPPORTED_ARCH ${X86} ${X86_64} ${ARM32} ${ARM64}
    ${MIPS32} ${MIPS64} ${WASM32} ${WASM64})

include(CompilerRTUtils)
include(CompilerRTDarwinUtils)

if(APPLE)

  find_darwin_sdk_dir(DARWIN_osx_SYSROOT macosx)
  find_darwin_sdk_dir(DARWIN_iossim_SYSROOT iphonesimulator)
  find_darwin_sdk_dir(DARWIN_ios_SYSROOT iphoneos)
  find_darwin_sdk_dir(DARWIN_watchossim_SYSROOT watchsimulator)
  find_darwin_sdk_dir(DARWIN_watchos_SYSROOT watchos)
  find_darwin_sdk_dir(DARWIN_tvossim_SYSROOT appletvsimulator)
  find_darwin_sdk_dir(DARWIN_tvos_SYSROOT appletvos)

  set(DARWIN_EMBEDDED_PLATFORMS)
  set(DARWIN_osx_BUILTIN_MIN_VER 10.5)
  set(DARWIN_osx_BUILTIN_MIN_VER_FLAG
      -mmacosx-version-min=${DARWIN_osx_BUILTIN_MIN_VER})

  if(COMPILER_RT_ENABLE_IOS)
    list(APPEND DARWIN_EMBEDDED_PLATFORMS ios)
    set(DARWIN_ios_MIN_VER_FLAG -miphoneos-version-min)
    set(DARWIN_ios_BUILTIN_MIN_VER 6.0)
    set(DARWIN_ios_BUILTIN_MIN_VER_FLAG
      ${DARWIN_ios_MIN_VER_FLAG}=${DARWIN_ios_BUILTIN_MIN_VER})
  endif()
  if(COMPILER_RT_ENABLE_WATCHOS)
    list(APPEND DARWIN_EMBEDDED_PLATFORMS watchos)
    set(DARWIN_watchos_MIN_VER_FLAG -mwatchos-version-min)
    set(DARWIN_watchos_BUILTIN_MIN_VER 2.0)
    set(DARWIN_watchos_BUILTIN_MIN_VER_FLAG
      ${DARWIN_watchos_MIN_VER_FLAG}=${DARWIN_watchos_BUILTIN_MIN_VER})
  endif()
  if(COMPILER_RT_ENABLE_TVOS)
    list(APPEND DARWIN_EMBEDDED_PLATFORMS tvos)
    set(DARWIN_tvos_MIN_VER_FLAG -mtvos-version-min)
    set(DARWIN_tvos_BUILTIN_MIN_VER 9.0)
    set(DARWIN_tvos_BUILTIN_MIN_VER_FLAG
      ${DARWIN_tvos_MIN_VER_FLAG}=${DARWIN_tvos_BUILTIN_MIN_VER})
  endif()

  set(BUILTIN_SUPPORTED_OS osx)

  # We're setting the flag manually for each target OS
  set(CMAKE_OSX_DEPLOYMENT_TARGET "")

  if(NOT DARWIN_osx_ARCHS)
    set(DARWIN_osx_ARCHS i386 x86_64 x86_64h)
  endif()

  set(DARWIN_sim_ARCHS i386 x86_64)
  set(DARWIN_device_ARCHS armv7 armv7s armv7k arm64)

  message(STATUS "OSX supported arches: ${DARWIN_osx_ARCHS}")
  foreach(arch ${DARWIN_osx_ARCHS})
    list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
    set(CAN_TARGET_${arch} 1)
  endforeach()

  # Need to build a 10.4 compatible libclang_rt
  set(DARWIN_10.4_SYSROOT ${DARWIN_osx_SYSROOT})
  set(DARWIN_10.4_BUILTIN_MIN_VER 10.4)
  set(DARWIN_10.4_BUILTIN_MIN_VER_FLAG
      -mmacosx-version-min=${DARWIN_10.4_BUILTIN_MIN_VER})
  set(DARWIN_10.4_SKIP_CC_KEXT On)
  darwin_test_archs(10.4 DARWIN_10.4_ARCHS i386 x86_64)
  message(STATUS "OSX 10.4 supported builtin arches: ${DARWIN_10.4_ARCHS}")
  if(DARWIN_10.4_ARCHS)
    # don't include the Haswell slice in the 10.4 compatibility library
    list(REMOVE_ITEM DARWIN_10.4_ARCHS x86_64h)
    list(APPEND BUILTIN_SUPPORTED_OS 10.4)
  endif()

  foreach(platform ${DARWIN_EMBEDDED_PLATFORMS})
    if(DARWIN_${platform}sim_SYSROOT)
      set(DARWIN_${platform}sim_BUILTIN_MIN_VER
        ${DARWIN_${platform}_BUILTIN_MIN_VER})
      set(DARWIN_${platform}sim_BUILTIN_MIN_VER_FLAG
        ${DARWIN_${platform}_BUILTIN_MIN_VER_FLAG})

      set(DARWIN_${platform}sim_SKIP_CC_KEXT On)

      set(test_arches ${DARWIN_sim_ARCHS})
      if(DARWIN_${platform}sim_ARCHS)
        set(test_arches DARWIN_${platform}sim_ARCHS)
      endif()

      darwin_test_archs(${platform}sim
        DARWIN_${platform}sim_ARCHS
        ${test_arches})
      message(STATUS "${platform} Simulator supported builtin arches: ${DARWIN_${platform}sim_ARCHS}")
      if(DARWIN_${platform}sim_ARCHS)
        list(APPEND BUILTIN_SUPPORTED_OS ${platform}sim)
      endif()
      foreach(arch ${DARWIN_${platform}sim_ARCHS})
        list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
        set(CAN_TARGET_${arch} 1)
      endforeach()
    endif()

    if(DARWIN_${platform}_SYSROOT)
      set(test_arches ${DARWIN_device_ARCHS})
      if(DARWIN_${platform}_ARCHS)
        set(test_arches DARWIN_${platform}_ARCHS)
      endif()

      darwin_test_archs(${platform}
        DARWIN_${platform}_ARCHS
        ${test_arches})
      message(STATUS "${platform} supported builtin arches: ${DARWIN_${platform}_ARCHS}")
      if(DARWIN_${platform}_ARCHS)
        list(APPEND BUILTIN_SUPPORTED_OS ${platform})
      endif()
      foreach(arch ${DARWIN_${platform}_ARCHS})
        list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
        set(CAN_TARGET_${arch} 1)
      endforeach()
    endif()
  endforeach()

  list_intersect(BUILTIN_SUPPORTED_ARCH ALL_BUILTIN_SUPPORTED_ARCH COMPILER_RT_SUPPORTED_ARCH)

else()
  # If we're not building the builtins standalone, just rely on the  tests in
  # config-ix.cmake to tell us what to build. Otherwise we need to do some leg
  # work here...
  if(COMPILER_RT_BUILTINS_STANDALONE_BUILD)
    test_targets()
  endif()
  # Architectures supported by compiler-rt libraries.
  filter_available_targets(BUILTIN_SUPPORTED_ARCH
    ${ALL_BUILTIN_SUPPORTED_ARCH})
endif()

message(STATUS "Builtin supported architectures: ${BUILTIN_SUPPORTED_ARCH}")
