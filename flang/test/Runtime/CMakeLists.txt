#===-- test/Runtime/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#------------------------------------------------------------------------------#

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()

add_executable(format-test
  format.cpp
)

target_link_libraries(format-test
  FortranRuntime
)

add_test(Format format-test)

add_executable(hello-world
  hello.cpp
)

target_link_libraries(hello-world
  FortranRuntime
)

add_test(HelloWorld hello-world)

add_executable(external-hello-world
  external-hello.cpp
)

target_link_libraries(external-hello-world
  FortranRuntime
)
