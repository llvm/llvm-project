cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(PACKAGE_VERSION "11.0.0git")

project(LLVM_RUNTIMES)

# Include individual projects
set(LLVM_ENABLE_PROJECTS "" CACHE STRING "Semicolon-separated list of runtimes to build.")

set(LLVM_UMBRELLA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
foreach(project IN LISTS LLVM_ENABLE_PROJECTS)
    add_subdirectory("${LLVM_UMBRELLA_ROOT}/${project}" "${CMAKE_CURRENT_BINARY_DIR}/${project}")
endforeach()
