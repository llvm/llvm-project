# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(CLANG_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/clang)
set(clang_cmake_builddir "${CMAKE_BINARY_DIR}/${CLANG_INSTALL_PACKAGE_DIR}")

get_property(CLANG_EXPORTS GLOBAL PROPERTY CLANG_EXPORTS)
export(TARGETS ${CLANG_EXPORTS} FILE ${clang_cmake_builddir}/ClangTargets.cmake)

# Generate ClangConfig.cmake for the build tree.
set(CLANG_CONFIG_CMAKE_DIR "${clang_cmake_builddir}")
set(CLANG_CONFIG_EXPORTS_FILE "${clang_cmake_builddir}/ClangTargets.cmake")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake.in
  ${clang_cmake_builddir}/ClangConfig.cmake
  @ONLY)
set(CLANG_CONFIG_CMAKE_DIR)
set(CLANG_CONFIG_EXPORTS_FILE)

# Generate ClangConfig.cmake for the install tree.
set(CLANG_CONFIG_CODE "
# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(CLANG_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${CLANG_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(CLANG_CONFIG_CODE "${CLANG_CONFIG_CODE}
get_filename_component(CLANG_INSTALL_PREFIX \"\${CLANG_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(CLANG_CONFIG_CMAKE_DIR "\${CLANG_INSTALL_PREFIX}/${CLANG_INSTALL_PACKAGE_DIR}")
set(CLANG_CONFIG_EXPORTS_FILE "\${CLANG_CMAKE_DIR}/ClangTargets.cmake")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfig.cmake
  @ONLY)
set(CLANG_CONFIG_CODE)
set(CLANG_CONFIG_CMAKE_DIR)
set(CLANG_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(EXPORT ClangTargets DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfig.cmake
    DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})
endif()
