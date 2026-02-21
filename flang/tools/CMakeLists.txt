#===-- tools/CMakeLists.txt ------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(bbc)
add_subdirectory(f18)
add_subdirectory(flang-driver)
add_subdirectory(tco)
add_subdirectory(f18-parse-demo)
add_subdirectory(fir-opt)
add_subdirectory(fir-lsp-server)
add_subdirectory(flang-tidy)

if(FLANG_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

option(FLANG_TOOLS_INCLUDE_DOCS "Generate build targets for the Flang Tools docs."
  ${LLVM_INCLUDE_DOCS})
if( FLANG_TOOLS_INCLUDE_DOCS )
  add_subdirectory(docs)
endif()