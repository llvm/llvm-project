cmake_minimum_required(VERSION 3.20.0)

project(DSMIL LANGUAGES C CXX)

include(GNUInstallDirs)
include(AddLLVM)

add_subdirectory(lib)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dsmil
  FILES_MATCHING
  PATTERN "*.h"
)

set(DSMIL_WRAPPER_TEMPLATES
  tools/dsmil-clang.in
  tools/dsmil-clang++.in
  tools/dsmil-opt.in
)

set(DSMIL_WRAPPER_OUTPUTS "")
foreach(wrapper ${DSMIL_WRAPPER_TEMPLATES})
  get_filename_component(wrapper_name ${wrapper} NAME_WE)
  set(configured_wrapper ${CMAKE_CURRENT_BINARY_DIR}/${wrapper_name})
  configure_file(${wrapper} ${configured_wrapper} @ONLY)
  list(APPEND DSMIL_WRAPPER_OUTPUTS ${configured_wrapper})
endforeach()

install(
  PROGRAMS ${DSMIL_WRAPPER_OUTPUTS}
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  FILES cmake/dsmil-toolchain.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dsmil
)
