# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(LLD_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/lld)
set(lld_cmake_builddir "${CMAKE_BINARY_DIR}/${LLD_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(LLD_EXPORTS GLOBAL PROPERTY LLD_EXPORTS)
export(TARGETS ${LLD_EXPORTS} FILE ${lld_cmake_builddir}/LLDTargets.cmake)

# Generate LLDConfig.cmake for the build tree.
set(LLD_CONFIG_CMAKE_DIR "${lld_cmake_builddir}")
set(LLD_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(LLD_CONFIG_INCLUDE_EXPORTS "include(\"${lld_cmake_builddir}/LLDTargets.cmake\")")
set(LLD_CONFIG_INCLUDE_DIRS
  "${LLD_SOURCE_DIR}/include"
  "${LLD_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/LLDConfig.cmake.in
  ${lld_cmake_builddir}/LLDConfig.cmake
  @ONLY)
set(LLD_CONFIG_CMAKE_DIR)
set(LLD_CONFIG_LLVM_CMAKE_DIR)

# Generate LLDConfig.cmake for the install tree.
set(LLD_CONFIG_CODE "
# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(LLD_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${LLD_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(LLD_CONFIG_CODE "${LLD_CONFIG_CODE}
get_filename_component(LLD_INSTALL_PREFIX \"\${LLD_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(LLD_CONFIG_CMAKE_DIR "\${LLD_INSTALL_PREFIX}/${LLD_INSTALL_PACKAGE_DIR}")
set(LLD_CONFIG_LLVM_CMAKE_DIR "\${LLD_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
get_config_exports_includes(LLD LLD_CONFIG_INCLUDE_EXPORTS)
set(LLD_CONFIG_INCLUDE_DIRS "\${LLD_INSTALL_PREFIX}/include")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/LLDConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LLDConfig.cmake
  @ONLY)
set(LLD_CONFIG_CODE)
set(LLD_CONFIG_CMAKE_DIR)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install_distribution_exports(LLD)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LLDConfig.cmake
    DESTINATION ${LLD_INSTALL_PACKAGE_DIR}
    COMPONENT lld-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(lld-cmake-exports)
    add_llvm_install_targets(install-lld-cmake-exports
                             COMPONENT lld-cmake-exports)
  endif()
endif()
