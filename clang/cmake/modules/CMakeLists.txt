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

# Generate ClangConfig.cmake for the build tree.
set(CLANG_CONFIG_CMAKE_DIR "${clang_cmake_builddir}")
set(CLANG_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
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

# ClangIR's MLIR dependency is excluded from the aggregate build and install.
# Promote and install only the MLIR targets reachable from Clang's SDK, along
# with the headers needed to consume those targets.
set(clang_config_needs_mlir OFF)
if(CLANG_ENABLE_CIR AND NOT "mlir" IN_LIST LLVM_DEPENDENCY_ONLY_PROJECTS)
  set(clang_config_needs_mlir ON)
endif()
function(clang_config_find_mlir out_var mlir_cmake_dir)
  if(clang_config_needs_mlir)
    set(${out_var}
"find_package(MLIR \${LLVM_VERSION_MAJOR}.\${LLVM_VERSION_MINOR}.\${LLVM_VERSION_PATCH} EXACT CONFIG QUIET
             HINTS \"${mlir_cmake_dir}\")
if(NOT MLIR_FOUND)
  string(CONCAT \${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
    \"This Clang was built with ClangIR (CLANG_ENABLE_CIR=ON) and its libraries \"
    \"link against MLIR, but MLIRConfig.cmake was not found. Expected it at \"
    \"${mlir_cmake_dir}. Install MLIR alongside Clang or set MLIR_DIR to \"
    \"<prefix>/lib/cmake/mlir.\")
  set(\${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endif()" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()
clang_config_find_mlir(CLANG_CONFIG_FIND_MLIR "${llvm_builddir}/cmake/mlir")
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
# Keep this in sync with mlir/cmake/modules/CMakeLists.txt!
set(MLIR_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/mlir" CACHE STRING
  "Path for CMake subdirectory for MLIR (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/mlir')")
extend_path(clang_config_mlir_cmake_dir "\${CLANG_INSTALL_PREFIX}" "${MLIR_INSTALL_PACKAGE_DIR}")
clang_config_find_mlir(CLANG_CONFIG_FIND_MLIR "${clang_config_mlir_cmake_dir}")
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
