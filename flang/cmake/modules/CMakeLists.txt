# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(FLANG_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/flang)
set(flang_cmake_builddir "${CMAKE_BINARY_DIR}/${FLANG_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(FLANG_EXPORTS GLOBAL PROPERTY FLANG_EXPORTS)
export(TARGETS ${FLANG_EXPORTS} FILE ${flang_cmake_builddir}/FlangTargets.cmake)

# Generate FlangConfig.cmake for the build tree.
set(FLANG_CONFIG_CMAKE_DIR "${flang_cmake_builddir}")
set(FLANG_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(FLANG_CONFIG_EXPORTS_FILE "${flang_cmake_builddir}/FlangTargets.cmake")
set(FLANG_CONFIG_INCLUDE_DIRS
  "${FLANG_SOURCE_DIR}/include"
  "${FLANG_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FlangConfig.cmake.in
  ${flang_cmake_builddir}/FlangConfig.cmake
  @ONLY)
set(FLANG_CONFIG_CMAKE_DIR)
set(FLANG_CONFIG_LLVM_CMAKE_DIR)
set(FLANG_CONFIG_EXPORTS_FILE)

# Generate FlangConfig.cmake for the install tree.
set(FLANG_CONFIG_CODE "
  # Compute the installation prefix from this LLVMConfig.cmake file location.
  get_filename_component(FLANG_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${FLANG_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(FLANG_CONFIG_CODE "${FLANG_CONFIG_CODE}
      get_filename_component(FLANG_INSTALL_PREFIX \"\${FLANG_INSTALL_PREFIX}\" PATH)")
endforeach(p)

set(FLANG_CONFIG_CMAKE_DIR "\${FLANG_INSTALL_PREFIX}/${FLANG_INSTALL_PACKAGE_DIR}")
set(FLANG_CONFIG_LLVM_CMAKE_DIR "\${FLANG_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(FLANG_CONFIG_EXPORTS_FILE "\${FLANG_CMAKE_DIR}/FlangTargets.cmake")
set(FLANG_CONFIG_INCLUDE_DIRS "\${FLANG_INSTALL_PREFIX}/include")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FlangConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FlangConfig.cmake
  @ONLY)

set(FLANG_CONFIG_CODE)
set(FLANG_CONFIG_CMAKE_DIR)
set(FLANG_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  get_property(flang_has_exports GLOBAL PROPERTY FLANG_HAS_EXPORTS)
  if(flang_has_exports)
    install(EXPORT FlangTargets DESTINATION ${FLANG_INSTALL_PACKAGE_DIR}
            COMPONENT flang-cmake-exports)
  endif()

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FlangConfig.cmake
    DESTINATION ${FLANG_INSTALL_PACKAGE_DIR}
    COMPONENT flang-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(flang-cmake-exports)
    add_llvm_install_targets(install-flang-cmake-exports
                             COMPONENT flang-cmake-exports)
  endif()
endif()
