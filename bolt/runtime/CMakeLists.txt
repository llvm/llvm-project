cmake_minimum_required(VERSION 3.13.4)
include(CheckIncludeFiles)

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
add_library(bolt_rt_hugify STATIC
  hugify.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )

set(BOLT_RT_FLAGS
  -ffreestanding
  -fno-exceptions
  -fno-rtti
  -fno-stack-protector
  -mno-sse)

# Don't let the compiler think it can create calls to standard libs
target_compile_options(bolt_rt_instr PRIVATE ${BOLT_RT_FLAGS} -fPIE)
target_include_directories(bolt_rt_instr PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(bolt_rt_hugify PRIVATE ${BOLT_RT_FLAGS})
target_include_directories(bolt_rt_hugify PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS bolt_rt_instr DESTINATION lib)
install(TARGETS bolt_rt_hugify DESTINATION lib)

if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*")
  add_library(bolt_rt_instr_osx STATIC
    instr.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )
  target_include_directories(bolt_rt_instr_osx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_options(bolt_rt_instr_osx PRIVATE
    -target x86_64-apple-darwin19.6.0
    ${BOLT_RT_FLAGS})
  install(TARGETS bolt_rt_instr_osx DESTINATION lib)
endif()
