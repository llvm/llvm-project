# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(CLANG_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/clang)
set(clang_cmake_builddir "${CMAKE_BINARY_DIR}/${CLANG_INSTALL_PACKAGE_DIR}")

get_property(CLANG_EXPORTS GLOBAL PROPERTY CLANG_EXPORTS)
export(TARGETS ${CLANG_EXPORTS} FILE ${clang_cmake_builddir}/ClangTargets.cmake)

# Generate ClangConfig.cmake for the build tree.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake.in
  ${clang_cmake_builddir}/ClangConfig.cmake
  @ONLY)

# Generate ClangConfig.cmake for the install tree.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfig.cmake
  @ONLY)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(EXPORT ClangTargets DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ClangConfig.cmake
    DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})
endif()
