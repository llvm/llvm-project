#===-- cmake/modules/AddFortranOffloadRuntime.txt --------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

macro(enable_cuda_compilation name files)
  if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "CUDA")
    if (BUILD_SHARED_LIBS)
      message(FATAL_ERROR
        "BUILD_SHARED_LIBS is not supported for CUDA build of Fortran runtime"
        )
    endif()

    enable_language(CUDA)

    # TODO: figure out how to make target property CUDA_SEPARABLE_COMPILATION
    # work, and avoid setting CMAKE_CUDA_SEPARABLE_COMPILATION.
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Treat all supported sources as CUDA files.
    set_source_files_properties(${files} PROPERTIES LANGUAGE CUDA)
    set(CUDA_COMPILE_OPTIONS)
    if ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang")
      # Allow varargs.
      set(CUDA_COMPILE_OPTIONS
        -Xclang -fcuda-allow-variadic-functions
        )
    endif()
    if ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA")
      set(CUDA_COMPILE_OPTIONS
        --expt-relaxed-constexpr
        # Disable these warnings:
        #   'long double' is treated as 'double' in device code
        -Xcudafe --diag_suppress=20208
        -Xcudafe --display_error_number
        )
    endif()
    set_source_files_properties(${files} PROPERTIES COMPILE_OPTIONS
      "${CUDA_COMPILE_OPTIONS}"
      )

    if (EXISTS "${FLANG_RT_LIBCUDACXX_PATH}/include")
      # When using libcudacxx headers files, we have to use them
      # for all files of F18 runtime.
      include_directories(AFTER ${FLANG_RT_LIBCUDACXX_PATH}/include)
      add_compile_definitions(RT_USE_LIBCUDACXX=1)
    endif()

    # Add an OBJECT library consisting of CUDA PTX.
    add_flangrt_library(${name}PTX OBJECT ${files})
    set_property(TARGET ${name}PTX PROPERTY CUDA_PTX_COMPILATION ON)
    if (FLANG_RT_CUDA_RUNTIME_PTX_WITHOUT_GLOBAL_VARS)
      target_compile_definitions(${name}PTX
        PRIVATE FLANGRT_NO_GLOBAL_VAR_DEFS
        )
    endif()
  endif()
endmacro()

macro(enable_omp_offload_compilation name files)
  if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "OpenMP")
    # OpenMP offload build only works with Clang compiler currently.

    if (BUILD_SHARED_LIBS)
      message(FATAL_ERROR
        "BUILD_SHARED_LIBS is not supported for OpenMP offload build of Fortran runtime"
        )
    endif()

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND
        "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")

      set(all_amdgpu_architectures
        "gfx700;gfx701;gfx801;gfx803;gfx900;gfx902;gfx906"
        "gfx908;gfx90a;gfx90c;gfx940;gfx1010;gfx1030"
        "gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036"
        "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151"
        "gfx1152"
        )
      set(all_nvptx_architectures
        "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62"
        "sm_70;sm_72;sm_75;sm_80;sm_86;sm_89;sm_90"
        )
      set(all_gpu_architectures
        "${all_amdgpu_architectures};${all_nvptx_architectures}"
        )
      # TODO: support auto detection on the build system.
      if (FLANG_RT_DEVICE_ARCHITECTURES STREQUAL "all")
        set(FLANG_RT_DEVICE_ARCHITECTURES ${all_gpu_architectures})
      endif()
      list(REMOVE_DUPLICATES FLANG_RT_DEVICE_ARCHITECTURES)

      string(REPLACE ";" "," compile_for_architectures
        "${FLANG_RT_DEVICE_ARCHITECTURES}"
        )

      set(OMP_COMPILE_OPTIONS
        -fopenmp
        -fvisibility=hidden
        -fopenmp-cuda-mode
        --offload-arch=${compile_for_architectures}
        # Force LTO for the device part.
        -foffload-lto
        )
      set_source_files_properties(${files} PROPERTIES COMPILE_OPTIONS
        "${OMP_COMPILE_OPTIONS}"
        )
      target_link_options(${name} PUBLIC ${OMP_COMPILE_OPTIONS})

      # Enable "declare target" in the source code.
      set_source_files_properties(${files}
        PROPERTIES COMPILE_DEFINITIONS OMP_OFFLOAD_BUILD
        )
    else()
      message(FATAL_ERROR
        "Flang-rt build with OpenMP offload is not supported for these compilers:\n"
        "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}\n"
        "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
    endif()
  endif()
endmacro()
