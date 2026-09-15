#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Header-only Boost.Math, from the vendored snapshot in third-party/boost-math.
# The snapshot ships its own CMakeLists.txt, but it is deliberately not
# add_subdirectory()'d: consumers need nothing from it beyond the include path and
# BOOST_MATH_STANDALONE. Declaring the interface here keeps the snapshot a pure
# header drop, marks the includes SYSTEM without patching upstream's build files,
# and gives every consumer -- in any of LLVM's CMake invocations -- the same target.
if(NOT TARGET boost_math)
  set(boost_math_path ${LLVM_THIRD_PARTY_DIR}/boost-math)
  if(EXISTS ${boost_math_path} AND IS_DIRECTORY ${boost_math_path})
    add_library(boost_math INTERFACE)
    target_include_directories(boost_math SYSTEM INTERFACE ${boost_math_path}/include)
    target_compile_definitions(boost_math INTERFACE BOOST_MATH_STANDALONE=1)
  else()
    message(FATAL_ERROR "Boost.Math snapshot not found at ${boost_math_path}, but is required to build Boost.Math consumers (is LLVM_THIRD_PARTY_DIR set?)")
  endif()
endif()
