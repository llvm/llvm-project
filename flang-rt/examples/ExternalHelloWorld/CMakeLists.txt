#===-- examples/ExternalHelloWorld/CMakeLists.txt --------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# This test is not run by default as it requires input.
add_llvm_example(external-hello-world
  external-hello.cpp
)

target_link_libraries(external-hello-world
  PRIVATE
    flang_rt.runtime.default
  )
