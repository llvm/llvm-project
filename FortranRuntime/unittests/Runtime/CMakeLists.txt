#===-- unittests/Runtime/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flang_unittest(FlangRuntimeTests
  AccessTest.cpp
  Allocatable.cpp
  ArrayConstructor.cpp
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
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)

add_subdirectory(CUDA)
