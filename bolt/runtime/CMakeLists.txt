cmake_minimum_required(VERSION 3.20.0)
include(CheckIncludeFiles)
include(GNUInstallDirs)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

project(libbolt_rt_project)

check_include_files(elf.h HAVE_ELF_H)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)

add_library(bolt_rt_instr STATIC
  instr.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )
set_target_properties(bolt_rt_instr PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_DIR}")
add_library(bolt_rt_hugify STATIC
  hugify.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )
set_target_properties(bolt_rt_hugify PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_DIR}")

set(BOLT_RT_FLAGS
  -ffreestanding
  -fno-exceptions
  -fno-rtti
  -fno-stack-protector
  -fPIC
  -mgeneral-regs-only)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(BOLT_RT_FLAGS ${BOLT_RT_FLAGS} "-mno-sse")
endif()

# Don't let the compiler think it can create calls to standard libs
target_compile_options(bolt_rt_instr PRIVATE ${BOLT_RT_FLAGS})
target_include_directories(bolt_rt_instr PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(bolt_rt_hugify PRIVATE ${BOLT_RT_FLAGS})
target_include_directories(bolt_rt_hugify PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS bolt_rt_instr DESTINATION "lib${LLVM_LIBDIR_SUFFIX}")
install(TARGETS bolt_rt_hugify DESTINATION "lib${LLVM_LIBDIR_SUFFIX}")

if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  add_library(bolt_rt_instr_osx STATIC
    instr.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )
  set_target_properties(bolt_rt_instr_osx PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_DIR}")
  target_include_directories(bolt_rt_instr_osx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_options(bolt_rt_instr_osx PRIVATE
    -target x86_64-apple-darwin19.6.0
    ${BOLT_RT_FLAGS})
  install(TARGETS bolt_rt_instr_osx DESTINATION "lib${LLVM_LIBDIR_SUFFIX}")
endif()
