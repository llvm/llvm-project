include(GNUInstallPackageDir)
include(ExtendPath)
include(FindPrefixFromConfig)

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(FLANG_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/flang" CACHE STRING
  "Path for CMake subdirectory for Flang (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/flang')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
set(flang_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/flang")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/llvm" CACHE STRING
  "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
set(llvm_cmake_builddir "${LLVM_LIBRARY_DIR}/cmake/llvm")

get_property(FLANG_EXPORTS GLOBAL PROPERTY FLANG_EXPORTS)
export(TARGETS ${FLANG_EXPORTS} FILE ${flang_cmake_builddir}/FlangTargets.cmake)

# Generate FlangConfig.cmake for the build tree.
set(FLANG_CONFIG_CMAKE_DIR "${flang_cmake_builddir}")
set(FLANG_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(FLANG_CONFIG_INCLUDE_EXPORTS "include(\"${flang_cmake_builddir}/FlangTargets.cmake\")")
set(FLANG_CONFIG_INCLUDE_DIRS
  "${FLANG_SOURCE_DIR}/include"
  "${FLANG_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FlangConfig.cmake.in
  ${flang_cmake_builddir}/FlangConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FlangConfigVersion.cmake.in
  ${flang_cmake_builddir}/FlangConfigVersion.cmake
  @ONLY)
set(FLANG_CONFIG_CMAKE_DIR)
set(FLANG_CONFIG_LLVM_CMAKE_DIR)

# Generate FlangConfig.cmake for the install tree.
find_prefix_from_config(FLANG_CONFIG_CODE FLANG_INSTALL_PREFIX "${FLANG_INSTALL_PACKAGE_DIR}")

extend_path(FLANG_CONFIG_CMAKE_DIR "\${FLANG_INSTALL_PREFIX}" "${FLANG_INSTALL_PACKAGE_DIR}")
extend_path(FLANG_CONFIG_LLVM_CMAKE_DIR "\${FLANG_INSTALL_PREFIX}" "${LLVM_INSTALL_PACKAGE_DIR}")
get_config_exports_includes(Flang FLANG_CONFIG_INCLUDE_EXPORTS)
extend_path(FLANG_CONFIG_INCLUDE_DIRS "\${FLANG_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FlangConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FlangConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FlangConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FlangConfigVersion.cmake
  @ONLY)

set(FLANG_CONFIG_CODE)
set(FLANG_CONFIG_CMAKE_DIR)
set(FLANG_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install_distribution_exports(Flang)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FlangConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FlangConfigVersion.cmake
    DESTINATION ${FLANG_INSTALL_PACKAGE_DIR}
    COMPONENT flang-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(flang-cmake-exports)
    add_llvm_install_targets(install-flang-cmake-exports
                             COMPONENT flang-cmake-exports)
  endif()
endif()
