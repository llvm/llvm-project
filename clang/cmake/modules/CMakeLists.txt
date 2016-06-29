# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(CLANG_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/clang)
set(clang_cmake_builddir "${CMAKE_BINARY_DIR}/${CLANG_INSTALL_PACKAGE_DIR}")
get_property(CLANG_EXPORTS GLOBAL PROPERTY CLANG_EXPORTS)
export(TARGETS ${CLANG_EXPORTS} FILE ${clang_cmake_builddir}/ClangTargets.cmake)

# Install a <prefix>/lib/cmake/clang/ClangConfig.cmake file so that
# find_package(Clang) works. Install the target list with it.
install(EXPORT ClangTargets DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake
  DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})

# Also copy ClangConfig.cmake to the build directory so that dependent projects
# can build against a build directory of Clang more easily.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ClangConfig.cmake
  ${CLANG_BINARY_DIR}/${CLANG_INSTALL_PACKAGE_DIR}/ClangConfig.cmake
  COPYONLY)
