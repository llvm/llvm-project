#===-- cmake/modules/FlangRTIntrospection.cmake ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#


# Check whether the Fortran compiler supports real(16)/quadmath types
#
# Implementation notes:
#  * FORTRAN_SUPPORTS_REAL16 can be set externally in a bootstrapping-runtimes
#    build to ensure consistency of real(16) support between compiler and
#    runtime.
#
#  * Does not work with Flang and CMake < 3.24
#
#  * This is intentionally wrapped in a function to get its own namespace for
#    CMAKE_REQUIRED_FLAGS and CMAKE_TRY_COMPILE_TARGET_TYPE. In particular,
#    cmake_pop_check_state() does not reset CMAKE_TRY_COMPILE_TARGET_TYPE,
#    causing later try_compile invocations to fail. If you see
#    enable_language(CUDA) failing because CMAKE_RANLIB is empty, this is the
#    reason.
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
