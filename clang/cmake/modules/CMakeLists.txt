include(GNUInstallPackageDir)
include(ExtendPath)
include(LLVMDistributionSupport)
include(FindPrefixFromConfig)

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(CLANG_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/clang" CACHE STRING
  "Path for CMake subdirectory for Clang (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/clang')")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/llvm" CACHE STRING
  "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
string(REPLACE "${CMAKE_CFG_INTDIR}" "." llvm_builddir "${LLVM_LIBRARY_DIR}")
set(llvm_cmake_builddir "${llvm_builddir}/cmake/llvm")
if(CLANG_BUILT_STANDALONE)
  # CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
  set(clang_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/clang")
else()
  set(clang_cmake_builddir "${llvm_builddir}/cmake/clang")
endif()

get_property(CLANG_EXPORTS GLOBAL PROPERTY CLANG_EXPORTS)
if(NOT "clang" IN_LIST LLVM_DEPENDENCY_ONLY_PROJECTS)
  export(TARGETS ${CLANG_EXPORTS} FILE ${clang_cmake_builddir}/ClangTargets.cmake)
endif()

# ClangIR links MLIR. How that dependency reaches a find_package(Clang)
# consumer depends on how MLIR was enabled in this build:
#
# - MLIR listed in LLVM_ENABLE_PROJECTS: MLIR installs its own full package and
#   is the sole owner of its exported targets. ClangTargets.cmake references
#   them without defining them, so ClangConfig.cmake must load the MLIR package
#   first. Re-exporting them from Clang would create two owners for the same
#   targets and break find_package(MLIR) in one import order or the other.
#
# - MLIR enabled implicitly as a ClangIR dependency, i.e. listed in
#   LLVM_DEPENDENCY_ONLY_PROJECTS: no MLIR package is installed.
#   clang/CMakeLists.txt promotes the reachable MLIR closure into
#   ClangTargets.cmake instead, so there is nothing to find.
#
# The lookup in ClangConfig.cmake is optional: with LLVM_DISTRIBUTION_COMPONENTS
# or LLVM_DISTRIBUTIONS, an installed package may contain only export sets
# that do not link MLIR, and the installed targets file itself reports any
# MLIR target it references but cannot find.
#
# Flang loads its Clang and MLIR dependencies the same way, with a REQUIRED
# lookup; Clang's is optional for the distribution reason above.
if(CLANG_ENABLE_CIR AND NOT "mlir" IN_LIST LLVM_DEPENDENCY_ONLY_PROJECTS)
  set(CLANG_CONFIG_FIND_MLIR ON)
  # llvm/tools/CMakeLists.txt configures MLIR before Clang, so MLIR's package
  # directory variables are already set here.
  set(clang_config_mlir_cmake_builddir "${llvm_builddir}/cmake/mlir")
  extend_path(clang_config_mlir_cmake_installdir "\${CLANG_INSTALL_PREFIX}" "${MLIR_INSTALL_PACKAGE_DIR}")
else()
  set(CLANG_CONFIG_FIND_MLIR OFF)
  set(clang_config_mlir_cmake_builddir "")
  set(clang_config_mlir_cmake_installdir "")
endif()

# Generate ClangConfig.cmake for the build tree.
set(CLANG_CONFIG_CMAKE_DIR "${clang_cmake_builddir}")
set(CLANG_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(CLANG_CONFIG_MLIR_CMAKE_DIR "${clang_config_mlir_cmake_builddir}")
if("clang" IN_LIST LLVM_DEPENDENCY_ONLY_PROJECTS)
  # Clang is enabled only to satisfy another project's dependency, so its
  # targets file is not exported above. Leave the include out of the config
  # rather than referring to a file that is never generated, which would make
  # every find_package(Clang) against this build tree fail.
  set(CLANG_CONFIG_INCLUDE_EXPORTS "")
else()
  set(CLANG_CONFIG_INCLUDE_EXPORTS "include(\"${clang_cmake_builddir}/ClangTargets.cmake\")")
endif()
set(CLANG_CONFIG_INCLUDE_DIRS
  "${CLANG_SOURCE_DIR}/include"
  "${CLANG_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake.in
  ${clang_cmake_builddir}/ClangConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfigVersion.cmake.in
  ${clang_cmake_builddir}/ClangConfigVersion.cmake
  @ONLY)
set(CLANG_CONFIG_CMAKE_DIR)
set(CLANG_CONFIG_LLVM_CMAKE_DIR)
set(CLANG_CONFIG_MLIR_CMAKE_DIR)

# For compatibility with projects that include(ClangConfig)
# via CMAKE_MODULE_PATH, place API modules next to it.
# Copy without source permissions because the source could be read-only,
# but we need to write into the copied folder.
file(COPY .
  DESTINATION ${clang_cmake_builddir}
  NO_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
  )

# Generate ClangConfig.cmake for the install tree.
find_prefix_from_config(CLANG_CONFIG_CODE CLANG_INSTALL_PREFIX "${CLANG_INSTALL_PACKAGE_DIR}")
extend_path(CLANG_CONFIG_CMAKE_DIR "\${CLANG_INSTALL_PREFIX}" "${CLANG_INSTALL_PACKAGE_DIR}")
extend_path(CLANG_CONFIG_LLVM_CMAKE_DIR "\${CLANG_INSTALL_PREFIX}" "${LLVM_INSTALL_PACKAGE_DIR}")
set(CLANG_CONFIG_MLIR_CMAKE_DIR "${clang_config_mlir_cmake_installdir}")
get_config_exports_includes(Clang CLANG_CONFIG_INCLUDE_EXPORTS)
extend_path(base_includedir "\${CLANG_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")
set(CLANG_CONFIG_INCLUDE_DIRS
  "${base_includedir}"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfigVersion.cmake
  @ONLY)
set(CLANG_CONFIG_CODE)
set(CLANG_CONFIG_CMAKE_DIR)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY AND
    NOT "clang" IN_LIST LLVM_DEPENDENCY_ONLY_PROJECTS)
  install_distribution_exports(Clang)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddClang.cmake
    DESTINATION ${CLANG_INSTALL_PACKAGE_DIR}
    COMPONENT clang-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(clang-cmake-exports)
    add_llvm_install_targets(install-clang-cmake-exports
                             COMPONENT clang-cmake-exports)
  endif()
endif()
