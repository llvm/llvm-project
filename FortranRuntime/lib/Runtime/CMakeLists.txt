#===-- lib/Runtime/CMakeLists.txt ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(Float128Math)

set(sources
  ../Common/binary-to-decimal.cpp
  ../Common/decimal-to-binary.cpp
  ISO_Fortran_binding.cpp
  allocator-registry.cpp
  allocatable.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  command.cpp
  complex-powi.cpp
  complex-reduction.c
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
  exceptions.cpp
  execute.cpp
  extensions.cpp
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
  iso_fortran_env_impl.f90
  iostat.cpp
  main.cpp
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
  random.cpp
  reduce.cpp
  reduction.cpp
  stat.cpp
  stop.cpp
  sum.cpp
  support.cpp
  temporary-stack.cpp
  terminator.cpp
  time-intrinsic.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit-map.cpp
  unit.cpp
  utf.cpp
)

set(public_headers "")
file(GLOB_RECURSE public_headers
  "${FLANGRUNTIME_SOURCE_DIR}/include/flang/Runtime/*.h"
  "${FLANGRUNTIME_SOURCE_DIR}/include/flang/Common/*.h"
  )

set(private_headers "")
file(GLOB_RECURSE private_headers
  "${FLANGRUNTIME_SOURCE_DIR}/lib/Runtime/*.h"
  "${FLANGRUNTIME_SOURCE_DIR}/lib/Common/*.h"
  )

include(AddFlangOffloadRuntime)

enable_cuda_compilation(FortranRuntime "${supported_files}")
enable_omp_offload_compilation("${supported_files}")

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
  list(APPEND sources ${f128_sources})
endif()

if (NOT WIN32)
  add_fortranruntime_library(FortranRuntime STATIC
    ${sources}
    INSTALL_WITH_TOOLCHAIN
    ADDITIONAL_HEADERS ${public_headers} ${private_headers}
  )

  # For unittests that link to FortranRuntime. Should link to the static version
  # of the library.
  add_library(FortranRuntime.static ALIAS FortranRuntime)
  add_library(FortranRuntime.unittest ALIAS FortranRuntime)
else()
  function (add_win_fortranruntime_library libtype suffix msvc_lib)
    set(name "FortranRuntime.${suffix}")
    add_fortranruntime_library(${name} ${libtype}
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
  endfunction ()

  add_win_fortranruntime_library(STATIC static     MultiThreaded)
  add_win_fortranruntime_library(STATIC static_dbg MultiThreadedDebug)

  # unittests link against LLVMSupport which is compiled with /MD
  add_win_fortranruntime_library(STATIC unittest   MultiThreadedDLL EXCLUDE_FROM_ALL)

  # FIXME: Before DLL versions of the runtime can be used, exported definitions
  # must be annotated with __declspec(dllexport).
  add_win_fortranruntime_library(SHARED dynamic     MultiThreadedDLL)
  add_win_fortranruntime_library(SHARED dynamic_dbg MultiThreadedDebugDLL)

  # Target for building all versions of the runtime
  add_custom_target(FortranRuntime)
  set_target_properties(FortranRuntime PROPERTIES FOLDER "Fortran Runtime/Meta")
  add_dependencies(FortranRuntime
      FortranRuntime.static
      FortranRuntime.static_dbg
      FortranRuntime.dynamic
      FortranRuntime.dynamic_dbg
    )
endif()

if (FORTRANRUNTIME_ENABLE_CUF)
  add_subdirectory(CUDA)
endif()
