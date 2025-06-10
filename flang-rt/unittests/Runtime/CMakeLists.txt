#===-- unittests/Runtime/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_unittest(RuntimeTests
  AccessTest.cpp
  Allocatable.cpp
  ArrayConstructor.cpp
  Assign.cpp
  BufferTest.cpp
  CharacterTest.cpp
  CommandTest.cpp
  Complex.cpp
  CrashHandlerFixture.cpp
  Derived.cpp
  ExternalIOTest.cpp
  Format.cpp
  Inquiry.cpp
  ListInputTest.cpp
  LogicalFormatTest.cpp
  Matmul.cpp
  MatmulTranspose.cpp
  MiscIntrinsic.cpp
  Namelist.cpp
  Numeric.cpp
  NumericalFormatTest.cpp
  Pointer.cpp
  Ragged.cpp
  Random.cpp
  Reduction.cpp
  RuntimeCrashTest.cpp
  Stop.cpp
  Support.cpp
  Time.cpp
  TemporaryStack.cpp
  Transformational.cpp

  LINK_LIBS
    flang_rt.runtime.unittest
)
target_compile_definitions(RuntimeTests PRIVATE NOT_EXE="${LLVM_TOOLS_DIR}/not${CMAKE_EXECUTABLE_SUFFIX}")

if (FLANG_RT_INCLUDE_CUF)
  add_subdirectory(CUDA)
endif ()
