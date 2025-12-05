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
