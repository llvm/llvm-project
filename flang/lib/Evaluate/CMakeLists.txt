include(CheckLibraryExists)

if (LIBPGMATH_DIR)
  # If pgmath library is found, it can be used for constant folding.
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    # pgmath uses _Complex, so only enable linking pgmath with flang in environments
    # that support it (MSVC is OK, pgmath uses _Fcomplex/_Dcomplex there).
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|MSVC")
      check_cxx_compiler_flag("-Werror -Wc99-extensions" HAS_WC99_EXTENSIONS_FLAG)
      if (HAS_WC99_EXTENSIONS_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-extensions")
      endif()
      add_compile_definitions(LINK_WITH_LIBPGMATH)
      message(STATUS "Found libpgmath: ${LIBPGMATH}")
    else()
      message(STATUS "Libpgmath will not be used because C99 complex is not supported.")
    endif()
  else()
    message(STATUS "Libpgmath not found in: ${LIBPGMATH_DIR}")
  endif()
endif()

check_library_exists(quadmath sinq "" FOUND_QUADMATH_LIB)
if (FLANG_INCLUDE_QUADMATH_H AND FOUND_QUADMATH_LIB)
  configure_file("${FLANG_SOURCE_DIR}/cmake/quadmath_wrapper.h.in" "${CMAKE_CURRENT_BINARY_DIR}/quadmath_wrapper.h")
  add_compile_definitions(HAS_QUADMATHLIB)
  set(QUADMATHLIB quadmath)
endif ()

add_flang_library(FortranEvaluate
  call.cpp
  character-value-impl.cpp
  character-value.cpp
  characteristics.cpp
  check-expression.cpp
  common.cpp
  complex-value.cpp
  constant.cpp
  expression.cpp
  fold.cpp
  fold-character.cpp
  fold-complex.cpp
  fold-designator.cpp
  fold-integer.cpp
  fold-logical.cpp
  fold-real.cpp
  fold-reduction.cpp
  formatting.cpp
  host.cpp
  initial-image.cpp
  integer.cpp
  integer-value.cpp
  intrinsics.cpp
  intrinsics-library.cpp
  logical-value.cpp
  real.cpp
  real-value.cpp
  shape.cpp
  static-data.cpp
  target.cpp
  tools.cpp
  type.cpp
  variable.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS
  FortranSupport
  FortranDecimal
  FortranParser
  ${LIBPGMATH}
  ${QUADMATHLIB}

  DISABLE_PCH_REUSE
  PRECOMPILE_HEADERS
  [["flang/Evaluate/common.h"]]
  [["flang/Evaluate/call.h"]]
  [["flang/Evaluate/traverse.h"]]
  [["flang/Evaluate/shape.h"]]
  [["flang/Evaluate/characteristics.h"]]
  [["flang/Evaluate/variable.h"]]
  [["flang/Evaluate/real-value.h"]]
  [["flang/Evaluate/type.h"]]
  [["flang/Evaluate/integer-value.h"]]
  [["flang/Evaluate/expression.h"]]
  [["flang/Evaluate/tools.h"]]

  LINK_COMPONENTS
  Support

  DEPENDS
  acc_gen
  omp_gen
)

# Automatic deduction of the opaque object size/alignment used by the RealValue
# and CharacterValue facades (real-value.h / character-value.h).  These embed a
# fixed-size, suitably
# aligned byte buffer standing in for their variant-backed implementation
# (RealValueImpl / CharacterValueImpl) so the implementation headers stay private.
# The required size/alignment is toolchain specific, so we compile a tiny probe
# with this very toolchain and let it emit the matching object-sizes.h into the
# build tree's include directory, which is searched before the source tree (see
# the include_directories(BEFORE ...) in flang/CMakeLists.txt).  The checked-in
# object-sizes.h then acts purely as a fallback, which is what gets used when
# cross-compiling (the probe cannot run on the host).
#
# The header is emitted per configuration ($<CONFIG>) and built by that
# configuration's probe, because multi-config generators may produce different
# object layouts per configuration.  $<CONFIG> in the OUTPUT/DEPENDS below is the
# documented Ninja Multi-Config pattern; on single-config generators it simply
# expands to the active build type.
if (NOT CMAKE_CROSSCOMPILING)
  add_executable(flang-evaluate-object-size-probe object-size-probe.cpp)
  set_target_properties(flang-evaluate-object-size-probe PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # The probe only needs sizeof/alignof from the headers; it does not link any
  # LLVM runtime, so opt out of the ABI-breaking-checks symbol enforcement that
  # would otherwise require linking LLVMSupport.
  #
  # FORTRAN_EVALUATE_OBJECT_SIZE_PROBE makes object-sizes.h ignore the generated
  # header that this probe produces, so the probe does not depend on its own
  # output (which would create a build-graph cycle).
  target_compile_definitions(flang-evaluate-object-size-probe PRIVATE
    LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1
    FORTRAN_EVALUATE_OBJECT_SIZE_PROBE)

  set(_object_sizes_dir
    "${FLANG_BINARY_DIR}/include/object-sizes/$<CONFIG>/flang/Evaluate")
  set(_object_sizes_h "${_object_sizes_dir}/object-sizes-generated.h")

  add_custom_command(
    OUTPUT "${_object_sizes_h}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_object_sizes_dir}"
    COMMAND "$<TARGET_FILE:flang-evaluate-object-size-probe>" "${_object_sizes_h}"
    DEPENDS flang-evaluate-object-size-probe
    COMMENT "Deducing RealValueImpl/CharacterValueImpl object size and alignment ($<CONFIG>)"
    VERBATIM)

  add_custom_target(flang-evaluate-object-size DEPENDS "${_object_sizes_h}")

  add_dependencies(FortranEvaluate flang-evaluate-object-size)
endif()
