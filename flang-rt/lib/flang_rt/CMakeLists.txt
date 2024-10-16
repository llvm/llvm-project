#===-- lib/flang_rt/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

include(AddFlangRTOffload)

# List of files that are buildable for all devices.
set(supported_sources
  ISO_Fortran_binding.cpp
  allocator-registry.cpp
  allocatable.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  connection.cpp
  copy.cpp
  derived-api.cpp
  derived.cpp
  descriptor-io.cpp
  descriptor.cpp
  dot-product.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  external-unit.cpp
  extrema.cpp
  file.cpp
  findloc.cpp
  format.cpp
  inquiry.cpp
  internal-unit.cpp
  io-api.cpp
  io-api-minimal.cpp
  io-error.cpp
  io-stmt.cpp
  iostat.cpp
  matmul-transpose.cpp
  matmul.cpp
  memory.cpp
  misc-intrinsic.cpp
  namelist.cpp
  non-tbp-dio.cpp
  numeric.cpp
  pointer.cpp
  product.cpp
  pseudo-unit.cpp
  ragged.cpp
  stat.cpp
  sum.cpp
  support.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit.cpp
  unit-map.cpp
  utf.cpp
)

# List of source not used for GPU offloading.
set(host_sources
  ${FLANG_SOURCE_DIR}/lib/Common/binary-to-decimal.cpp
  ${FLANG_SOURCE_DIR}/lib/Common/decimal-to-binary.cpp
  command.cpp
  complex-powi.cpp
  complex-reduction.c
  exceptions.cpp
  execute.cpp
  extensions.cpp
  ${FLANG_SOURCE_DIR}/module/iso_fortran_env_impl.f90
  main.cpp
  random.cpp
  reduce.cpp
  reduction.cpp
  stop.cpp
  temporary-stack.cpp
  time-intrinsic.cpp
)

file(GLOB_RECURSE public_headers
  "${FLANGRT_SOURCE_DIR}/include/flang/Runtime/*.h"
  "${FLANGRT_SOURCE_DIR}/include/flang/Common/*.h"
  )

file(GLOB_RECURSE private_headers
  "${FLANGRT_SOURCE_DIR}/lib/Runtime/*.h"
  "${FLANGRT_SOURCE_DIR}/lib/Common/*.h"
  )


# Import changes from sibling FortranFloat128Math
get_target_property(f128_sources
  FortranFloat128MathILib INTERFACE_SOURCES
  )
if (f128_sources)
  # The interface may define special macros for Float128Math files,
  # so we need to propagate them.
  get_target_property(f128_defs
    FortranFloat128MathILib INTERFACE_COMPILE_DEFINITIONS
    )
  set_property(SOURCE ${f128_sources}
    APPEND PROPERTY COMPILE_DEFINITIONS
    ${f128_defs}
    )
else ()
  set(f128_sources "")
endif ()


set(sources ${supported_sources} ${host_sources} ${f128_sources})

if (NOT WIN32)
  add_flangrt_library(flang_rt STATIC
    ${sources}
    INSTALL_WITH_TOOLCHAIN
    ADDITIONAL_HEADERS ${public_headers} ${private_headers}
  )

  enable_cuda_compilation(flang_rt "${supported_sources}")
  enable_omp_offload_compilation("${supported_sources}")

  # For unittests that depend on flang_rt. Should link to the static version
  # of the library.
  add_library(flang_rt.static ALIAS flang_rt)
  add_library(flang_rt.unittest ALIAS flang_rt)
else()
  function (add_win_flangrt_library libtype suffix msvc_lib)
    set(name "flang_rt.${suffix}")
    add_flangrt_library(${name} ${libtype}
        ${sources}
        ${ARGN}
        INSTALL_WITH_TOOLCHAIN
        ADDITIONAL_HEADERS ${public_headers} ${private_headers}
      )

    # Setting an unique Fortran_MODULE_DIRECTORY is required for each variant to
    # write a different .mod file.
    set_target_properties(${name}
        PROPERTIES
          MSVC_RUNTIME_LIBRARY ${msvc_lib}
          Fortran_MODULE_DIRECTORY "module.${suffix}"
      )

    enable_cuda_compilation(${name} "${supported_sources}")
    enable_omp_offload_compilation("${supported_sources}")
  endfunction ()

  add_win_flangrt_library(STATIC static     MultiThreaded)
  add_win_flangrt_library(STATIC static_dbg MultiThreadedDebug)

  # unittests link against LLVMSupport which is compiled with /MD
  add_win_flangrt_library(STATIC unittest   MultiThreadedDLL EXCLUDE_FROM_ALL)

  # FIXME: Before DLL versions of the runtime can be used, exported definitions
  # must be annotated with __declspec(dllexport).
  add_win_flangrt_library(SHARED dynamic     MultiThreadedDLL)
  add_win_flangrt_library(SHARED dynamic_dbg MultiThreadedDebugDLL)

  # Target for building all versions of the runtime
  add_custom_target(flang_rt)
  set_target_properties(flang_rt PROPERTIES FOLDER "Fortran Runtime/Meta")
  add_dependencies(flang_rt
      flang_rt.static
      flang_rt.static_dbg
      flang_rt.dynamic
      flang_rt.dynamic_dbg
    )
endif()
