function(_get_compile_options_from_flags output_var)
  set(compile_options "")

  if(LIBC_TARGET_ARCHITECTURE_IS_RISCV64 OR(LIBC_CPU_FEATURES MATCHES "FMA"))
    check_flag(ADD_FMA_FLAG ${FMA_OPT_FLAG} ${flags})
  endif()
  check_flag(ADD_SSE4_2_FLAG ${ROUND_OPT_FLAG} ${flags})
  check_flag(ADD_EXPLICIT_SIMD_OPT_FLAG ${EXPLICIT_SIMD_OPT_FLAG} ${flags})
  
  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    if(ADD_FMA_FLAG)
      if(LIBC_TARGET_ARCHITECTURE_IS_X86)
        list(APPEND compile_options "-mavx2")
        list(APPEND compile_options "-mfma")
      elseif(LIBC_TARGET_ARCHITECTURE_IS_RISCV64)
        list(APPEND compile_options "-D__LIBC_RISCV_USE_FMA")
      endif()
    endif()
    if(ADD_SSE4_2_FLAG)
      list(APPEND compile_options "-msse4.2")
    endif()
    if(ADD_EXPLICIT_SIMD_OPT_FLAG)
      list(APPEND compile_options "-D__LIBC_EXPLICIT_SIMD_OPT")
    endif()
  elseif(MSVC)
    if(ADD_FMA_FLAG)
      list(APPEND compile_options "/arch:AVX2")
    endif()
    if(ADD_EXPLICIT_SIMD_OPT_FLAG)
      list(APPEND compile_options "/D__LIBC_EXPLICIT_SIMD_OPT")
    endif()
  endif()

  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction(_get_compile_options_from_flags)

function(_get_common_compile_options output_var flags)
  _get_compile_options_from_flags(compile_flags ${flags})

  set(compile_options ${LIBC_COMPILE_OPTIONS_DEFAULT} ${compile_flags})

  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    list(APPEND compile_options "-fpie")

    if(LLVM_LIBC_FULL_BUILD)
      # Only add -ffreestanding flag in full build mode.
      list(APPEND compile_options "-ffreestanding")
    endif()

    if(LIBC_COMPILER_HAS_FIXED_POINT)
      list(APPEND compile_options "-ffixed-point")
    endif()

    list(APPEND compile_options "-fno-builtin")
    list(APPEND compile_options "-fno-exceptions")
    list(APPEND compile_options "-fno-lax-vector-conversions")
    list(APPEND compile_options "-fno-unwind-tables")
    list(APPEND compile_options "-fno-asynchronous-unwind-tables")
    list(APPEND compile_options "-fno-rtti")
    if (LIBC_CC_SUPPORTS_PATTERN_INIT)
      list(APPEND compile_options "-ftrivial-auto-var-init=pattern")
    endif()
    list(APPEND compile_options "-Wall")
    list(APPEND compile_options "-Wextra")
    # -DLIBC_WNO_ERROR=ON if you can't build cleanly with -Werror.
    if(NOT LIBC_WNO_ERROR)
      list(APPEND compile_options "-Werror")
    endif()
    list(APPEND compile_options "-Wconversion")
    list(APPEND compile_options "-Wno-sign-conversion")
    list(APPEND compile_options "-Wimplicit-fallthrough")
    list(APPEND compile_options "-Wwrite-strings")
    list(APPEND compile_options "-Wextra-semi")
    if(NOT CMAKE_COMPILER_IS_GNUCXX)
      list(APPEND compile_options "-Wnewline-eof")
      list(APPEND compile_options "-Wnonportable-system-include-path")
      list(APPEND compile_options "-Wstrict-prototypes")
      list(APPEND compile_options "-Wthread-safety")
      list(APPEND compile_options "-Wglobal-constructors")
    endif()
  elseif(MSVC)
    list(APPEND compile_options "/EHs-c-")
    list(APPEND compile_options "/GR-")
  endif()
  if (LIBC_TARGET_ARCHITECTURE_IS_GPU)
    list(APPEND compile_options "-nogpulib")
    list(APPEND compile_options "-fvisibility=hidden")
    list(APPEND compile_options "-fconvergent-functions")

    # Manually disable all standard include paths and include the resource
    # directory to prevent system headers from being included.
    list(APPEND compile_options "-isystem${COMPILER_RESOURCE_DIR}/include")
    list(APPEND compile_options "-nostdinc")
  endif()
  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction()

function(_get_common_test_compile_options output_var flags)
  _get_compile_options_from_flags(compile_flags ${flags})

  set(compile_options ${LIBC_COMPILE_OPTIONS_DEFAULT} ${compile_flags})

  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    list(APPEND compile_options "-fpie")

    if(LLVM_LIBC_FULL_BUILD)
      # Only add -ffreestanding flag in full build mode.
      list(APPEND compile_options "-ffreestanding")
      list(APPEND compile_options "-fno-exceptions")
      list(APPEND compile_options "-fno-unwind-tables")
      list(APPEND compile_options "-fno-asynchronous-unwind-tables")
      list(APPEND compile_options "-fno-rtti")
    endif()

    if(LIBC_COMPILER_HAS_FIXED_POINT)
      list(APPEND compile_options "-ffixed-point")
    endif()

    # list(APPEND compile_options "-Wall")
    # list(APPEND compile_options "-Wextra")
    # -DLIBC_WNO_ERROR=ON if you can't build cleanly with -Werror.
    if(NOT LIBC_WNO_ERROR)
      # list(APPEND compile_options "-Werror")
    endif()
    # list(APPEND compile_options "-Wconversion")
    # list(APPEND compile_options "-Wno-sign-conversion")
    # list(APPEND compile_options "-Wimplicit-fallthrough")
    # list(APPEND compile_options "-Wwrite-strings")
    # list(APPEND compile_options "-Wextra-semi")
    # if(NOT CMAKE_COMPILER_IS_GNUCXX)
    #   list(APPEND compile_options "-Wnewline-eof")
    #   list(APPEND compile_options "-Wnonportable-system-include-path")
    #   list(APPEND compile_options "-Wstrict-prototypes")
    #   list(APPEND compile_options "-Wthread-safety")
    #   list(APPEND compile_options "-Wglobal-constructors")
    # endif()
  endif()
  if (LIBC_TARGET_ARCHITECTURE_IS_GPU)
    # TODO: Set these flags
    # list(APPEND compile_options "-nogpulib")
    # list(APPEND compile_options "-fvisibility=hidden")
    # list(APPEND compile_options "-fconvergent-functions")

    # # Manually disable all standard include paths and include the resource
    # # directory to prevent system headers from being included.
    # list(APPEND compile_options "-isystem${COMPILER_RESOURCE_DIR}/include")
    # list(APPEND compile_options "-nostdinc")
  endif()
  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction()


# Obtains NVPTX specific arguments for compilation.
# The PTX feature is primarily based on the CUDA toolchain version. We want to
# be able to target NVPTX without an existing CUDA installation, so we need to
# set this manually. This simply sets the PTX feature to the minimum required
# for the features we wish to use on that target. The minimum PTX features used
# here roughly corresponds to the CUDA 9.0 release.
# Adjust as needed for desired PTX features.
function(get_nvptx_compile_options output_var gpu_arch)
  set(nvptx_options "")
  list(APPEND nvptx_options "-march=${gpu_arch}")
  list(APPEND nvptx_options "-Wno-unknown-cuda-version")
  list(APPEND nvptx_options "SHELL:-mllvm -nvptx-emit-init-fini-kernel=false")
  if(${gpu_arch} STREQUAL "sm_35")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_37")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_50")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_52")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_53")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_60")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_61")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_62")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_70")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_72")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_75")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_80")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  elseif(${gpu_arch} STREQUAL "sm_86")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  elseif(${gpu_arch} STREQUAL "sm_89")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  elseif(${gpu_arch} STREQUAL "sm_90")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  else()
    message(FATAL_ERROR "Unknown Nvidia GPU architecture '${gpu_arch}'")
  endif()

  if(LIBC_CUDA_ROOT)
    list(APPEND nvptx_options "--cuda-path=${LIBC_CUDA_ROOT}")
  endif()
  set(${output_var} ${nvptx_options} PARENT_SCOPE)
endfunction()

#TODO: Fold this into a function to get test framework compile options (which
# need to be separate from the main test compile options because otherwise they
# error)
set(LIBC_HERMETIC_TEST_COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_DEFAULT}
    -fpie -ffreestanding -fno-exceptions -fno-rtti)
# The GPU build requires overriding the default CMake triple and architecture.
if(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
  list(APPEND LIBC_HERMETIC_TEST_COMPILE_OPTIONS
       -nogpulib -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE} -flto
       --target=${LIBC_GPU_TARGET_TRIPLE}
       -mcode-object-version=${LIBC_GPU_CODE_OBJECT_VERSION})
elseif(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  get_nvptx_compile_options(nvptx_options ${LIBC_GPU_TARGET_ARCHITECTURE})
  list(APPEND LIBC_HERMETIC_TEST_COMPILE_OPTIONS
       -nogpulib ${nvptx_options} -fno-use-cxa-atexit --target=${LIBC_GPU_TARGET_TRIPLE})
endif()
