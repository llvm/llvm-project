# This file configures the CMake package configuration file "AIIRConfig.cmake"
# (see https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html).
# Two different versions of the package config file are generated and copied
# into different locations:
# 1. One in a canonical location in the build directory,
#   '<build>/lib/cmake/aiir/AIIRConfig.cmake'. This version may use absolute
#    paths referring to locations in the build tree.
# 2. One under '${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles', which is the version
#    *to be installed* when `cmake --install` command is run or when CPack
#    is invoked. This is the version that is distributed with LLVM packages.
#    The key difference between this and (1) is that all paths must be
#    relative the the installed "AIIRConfig.cmake" file.

include(GNUInstallPackageDir)
include(ExtendPath)
include(LLVMDistributionSupport)
include(FindPrefixFromConfig)

# This is a relative path to the location of AIIRConfig.cmake within the
# installation.
set(AIIR_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/aiir" CACHE STRING
  "Path for CMake subdirectory for AIIR (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/aiir')")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/llvm" CACHE STRING
  "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')")
string(REPLACE "${CMAKE_CFG_INTDIR}" "." llvm_cmake_builddir "${LLVM_LIBRARY_DIR}")

# These two variables contain absolute paths, so only use them for the
# build directory 'AIIRConfig.cmake'.
set(aiir_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/aiir")
set(llvm_cmake_builddir "${llvm_cmake_builddir}/cmake/llvm")

# -----------------------------------------------------------------------------
# Generate `<build>/lib/cmake/aiir/AIIRConfig.cmake'
# -----------------------------------------------------------------------------

# Generate the 'AIIRTarget.cmake' file.
get_property(AIIR_EXPORTS GLOBAL PROPERTY AIIR_EXPORTS)
export(TARGETS ${AIIR_EXPORTS} FILE ${aiir_cmake_builddir}/AIIRTargets.cmake)

# Declare all the variables required by the 'AIIRConfig.cmake.in' template.
get_property(AIIR_ALL_LIBS GLOBAL PROPERTY AIIR_ALL_LIBS)
get_property(AIIR_DIALECT_LIBS GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(AIIR_CONVERSION_LIBS GLOBAL PROPERTY AIIR_CONVERSION_LIBS)
get_property(AIIR_EXTENSION_LIBS GLOBAL PROPERTY AIIR_EXTENSION_LIBS)
get_property(AIIR_TRANSLATION_LIBS GLOBAL PROPERTY AIIR_TRANSLATION_LIBS)
get_property(AIIR_CAPI_LIBS GLOBAL PROPERTY AIIR_CAPI_LIBS)

# Generate AiirConfig.cmake for the build tree.
set(AIIR_CONFIG_CMAKE_DIR "${aiir_cmake_builddir}")
set(AIIR_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(AIIR_CONFIG_INCLUDE_EXPORTS "include(\"\${AIIR_CMAKE_DIR}/AIIRTargets.cmake\")")
set(AIIR_CONFIG_INCLUDE_DIRS
  "${AIIR_SOURCE_DIR}/include"
  "${AIIR_BINARY_DIR}/include"
  )
# Refer to the best host aiir-tblgen, which might be a host-optimized version
set(AIIR_CONFIG_TABLEGEN_EXE "${AIIR_TABLEGEN_EXE}")
set(AIIR_CONFIG_PDLL_TABLEGEN_EXE "${AIIR_PDLL_TABLEGEN_EXE}")
set(AIIR_CONFIG_SRC_SHARDER_TABLEGEN_EXE "${AIIR_SRC_SHARDER_TABLEGEN_EXE}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AIIRConfig.cmake.in
  ${aiir_cmake_builddir}/AIIRConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AIIRConfigVersion.cmake.in
  ${aiir_cmake_builddir}/AIIRConfigVersion.cmake
  @ONLY)

# Reset the variables which are invalid for the installed version of
# AIIRConfig.cmake.
set(AIIR_CONFIG_CMAKE_DIR)
set(AIIR_CONFIG_LLVM_CMAKE_DIR)
set(AIIR_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that add '<build>/lib/cmake/aiir' to
# their CMAKE_MODULE_PATH, place API modules next to it.
# Copy without source permissions because the source could be read-only,
# but we need to write into the copied folder.
file(COPY .
  DESTINATION ${aiir_cmake_builddir}
  NO_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
  )

# -----------------------------------------------------------------------------
# Generate AIIRConfig.cmake for the install tree.
# All path variables used in the template must be relative!
# -----------------------------------------------------------------------------

# This is a helper that sets 'AIIR_CONFIG_CODE' to some CMake code for
# insertion at the top of the generated 'AIIRConfig.cmake'. The code calculates
# the install directory prefix relative to the installed AIIRConfig.cmake (whose
# prefix is AIIR_INSTALL_PACKAGE_DIR), avoiding any use of absolute paths.
find_prefix_from_config(AIIR_CONFIG_CODE AIIR_INSTALL_PREFIX "${AIIR_INSTALL_PACKAGE_DIR}")

# Generate the other required paths by setting them relative to the
# variable 'AIIR_INSTALL_PREFIX'.
extend_path(AIIR_CONFIG_CMAKE_DIR "\${AIIR_INSTALL_PREFIX}" "${AIIR_INSTALL_PACKAGE_DIR}")
extend_path(AIIR_CONFIG_LLVM_CMAKE_DIR "\${AIIR_INSTALL_PREFIX}" "${LLVM_INSTALL_PACKAGE_DIR}")
get_config_exports_includes(AIIR AIIR_CONFIG_INCLUDE_EXPORTS)
extend_path(base_includedir "\${AIIR_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")
set(AIIR_CONFIG_INCLUDE_DIRS
  "${base_includedir}"
  )
# Ensure that we are using the installed aiir-tblgen. This might not be AIIR_TABLEGEN_EXE
# if we're building with a host-optimized aiir-tblgen (with LLVM_OPTIMIZED_TABLEGEN).
set(AIIR_CONFIG_TABLEGEN_EXE aiir-tblgen)
set(AIIR_CONFIG_PDLL_TABLEGEN_EXE aiir-pdll)
set(AIIR_CONFIG_SRC_SHARDER_TABLEGEN_EXE aiir-src-sharder)

set(AIIR_CONFIG_IRDL_TO_CPP_EXE aiir-irdl-to-cpp)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AIIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIIRConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AIIRConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIIRConfigVersion.cmake
  @ONLY)
set(AIIR_CONFIG_CODE)
set(AIIR_CONFIG_CMAKE_DIR)
set(AIIR_CONFIG_LLVM_CMAKE_DIR)
set(AIIR_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the AIIR parts as well
  # Include the cmake files so other tools can use aiir-tblgen, etc.
  install_distribution_exports(AIIR)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIIRConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIIRConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddAIIR.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddAIIRPython.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/IRDLToCpp.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AIIRDetectPythonEnv.cmake
    DESTINATION ${AIIR_INSTALL_PACKAGE_DIR}
    COMPONENT aiir-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(aiir-cmake-exports)
    add_llvm_install_targets(install-aiir-cmake-exports
                             COMPONENT aiir-cmake-exports)
  endif()
endif()
