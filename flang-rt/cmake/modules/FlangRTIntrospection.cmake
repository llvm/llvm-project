#===-- cmake/modules/FlangRTIntrospection.cmake ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

include(CMakePushCheckState)

# Check whether the Fortran compiler supports real(16)/quadmath types
#
# Implementation notes:
#
#  * FORTRAN_SUPPORTS_REAL16 can be set externally in a bootstrapping-runtimes
#    build to ensure consistency of real(16) support between compiler and
#    runtime.
#
#  * cmake_push_check_state/cmake_pop_check_state is insufficient to isolate
#    a compiler introspection environment, see
#    https://gitlab.kitware.com/cmake/cmake/-/issues/27419
#    Additionally wrap it in a function namespace.
function (check_fortran_quadmath_support)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "-ffree-form")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY") # Skip link step
  check_fortran_source_compiles([[
      subroutine test_quadmath
        real(16) :: var1
      end
    ]]
    FORTRAN_SUPPORTS_REAL16
  )
  cmake_pop_check_state()
endfunction ()

