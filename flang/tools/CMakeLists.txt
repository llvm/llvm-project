#===-- tools/CMakeLists.txt ------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(bbc)
add_subdirectory(flang-driver)
if (FLANG_STANDALONE_BUILD)
  # Mock flang only needed in bootstrapping builds;
  # fall back to full flang otherwise
  add_custom_target(flang-lazy)
  add_dependencies(flang-lazy flang)
else ()
  add_subdirectory(fakeflang)
endif ()
add_subdirectory(tco)
add_subdirectory(f18-parse-demo)
add_subdirectory(fir-opt)
add_subdirectory(fir-lsp-server)
