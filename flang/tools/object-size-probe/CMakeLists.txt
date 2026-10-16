#===-- tools/object-size-probe/CMakeLists.txt ------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

if (CMAKE_CROSSCOMPILING)
  # Cannot execute the probe when cross-compiling
  # Unlike tblgen, building it separately is no solution because we need the
  # sizeof(...)/alignof(...) to be processed for the target, not the host.
  # include/flang/Evaluate/object-sizes.h contains backup values when the
  # generated header does not exist
  return ()
endif ()


set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(flang-object-size-probe object-size-probe.cpp)
target_include_directories(flang-object-size-probe PRIVATE
  "${FLANG_SOURCE_DIR}/lib/Evaluate"
  )

set(_object_sizes_dir "${FLANG_BINARY_DIR}/include/object-sizes/$<CONFIG>/flang/Evaluate")
set(_object_sizes_h "${_object_sizes_dir}/object-sizes-generated.h")

add_custom_command(
    OUTPUT "${_object_sizes_h}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_object_sizes_dir}"
    COMMAND "$<TARGET_FILE:flang-object-size-probe>" "--write-if-changed" "-o" "${_object_sizes_h}"
    DEPENDS flang-object-size-probe
    COMMENT "Deducing IntegerValueImpl/RealValueImpl/CharacterValueImpl object size and alignment ($<CONFIG>)"
    VERBATIM)
add_custom_target(flang-generated-object-sizes DEPENDS "${_object_sizes_h}")

add_dependencies(FortranEvaluate flang-generated-object-sizes)
add_dependencies(FortranSemantics flang-generated-object-sizes)
add_dependencies(FortranLower flang-generated-object-sizes)
