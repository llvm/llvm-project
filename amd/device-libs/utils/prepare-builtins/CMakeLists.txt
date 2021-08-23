##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)

include(AddLLVM)

if (ROCM_DEVICELIB_STANDALONE_BUILD)
  add_definitions(${LLVM_DEFINITIONS})
  include_directories(${LLVM_INCLUDE_DIR})
  include_directories(${LLVM_CONFIG_INCLUDE_DIR})
  include_directories(${LLVM_MAIN_INCLUDE_DIR})

  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  link_directories("${LLVM_LIBRARY_DIR}")
endif()

add_executable(prepare-builtins prepare-builtins.cpp)
set_target_properties(prepare-builtins PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No)
llvm_update_compile_flags(prepare-builtins)
llvm_map_components_to_libnames(llvm_libs support core bitreader bitwriter)
target_link_libraries(prepare-builtins ${llvm_libs})
