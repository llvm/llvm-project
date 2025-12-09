#===-- cmake/modules/AddFlangRTOffload.cmake -------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

macro(enable_cuda_compilation name files)
  if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "CUDA")
    if (FLANG_RT_ENABLE_SHARED)
      message(FATAL_ERROR
        "FLANG_RT_ENABLE_SHARED is not supported for CUDA offload build of Flang-RT"
        )
    endif()

    enable_language(CUDA)

    set_target_properties(${name}.static
        PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
      )

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
      "${CUDA_COMPILE_OPTIONS}")

    # Create a .a library consisting of CUDA PTX.
    # This is different from a regular static library. The CUDA_PTX_COMPILATION
    # property can only be applied to object libraries and create *.ptx files
    # instead of *.o files. The .a will consist of those *.ptx files only.
    add_flangrt_library(obj.${name}PTX OBJECT ${files})
    set_target_properties(obj.${name}PTX PROPERTIES
      CUDA_PTX_COMPILATION ON
      CUDA_SEPARABLE_COMPILATION ON
      )
    add_flangrt_library(${name}PTX STATIC "$<TARGET_OBJECTS:obj.${name}PTX>")

    # Apply configuration options
    if (FLANG_RT_CUDA_RUNTIME_PTX_WITHOUT_GLOBAL_VARS)
      target_compile_definitions(obj.${name}PTX
        PRIVATE FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS
        )
    endif()

    # When using libcudacxx headers files, we have to use them
    # for all files of Flang-RT.
    if (EXISTS "${FLANG_RT_LIBCUDACXX_PATH}/include")
      foreach (tgt IN ITEMS "${name}.static" "obj.${name}PTX")
        target_include_directories(${tgt} AFTER PRIVATE "${FLANG_RT_LIBCUDACXX_PATH}/include")
        target_compile_definitions(${tgt} PRIVATE RT_USE_LIBCUDACXX=1)
      endforeach ()
    endif ()
  endif()
endmacro()

macro(enable_omp_offload_compilation name files)
  if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "OpenMP")
    # OpenMP offload build only works with Clang compiler currently.

    if (FLANG_RT_ENABLE_SHARED)
      message(FATAL_ERROR
        "FLANG_RT_ENABLE_SHARED is not supported for OpenMP offload build of Flang-RT"
        )
    endif()

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND
        "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")

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
      target_link_options(${name}.static PUBLIC ${OMP_COMPILE_OPTIONS})

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
