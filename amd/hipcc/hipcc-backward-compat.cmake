# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.16.8)

set(HIPCC_WRAPPER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/wrapper_dir/bin)
set(HIPCC_SRC_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${HIPCC_WRAPPER_BIN_DIR})
  #get all  binaries
  file(GLOB binary_files ${HIPCC_SRC_BIN_DIR}/*)
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${HIPCC_WRAPPER_BIN_DIR}/${file_name})
  endforeach()
endfunction()

# Create symlink to binaries
create_binary_symlink()
# TODO: Following has to modified if component based installation is required
if (NOT WIN32)
  install(DIRECTORY ${HIPCC_WRAPPER_BIN_DIR} DESTINATION hip)
else()
  install(DIRECTORY ${HIPCC_WRAPPER_BIN_DIR} DESTINATION hip
          FILES_MATCHING
          PATTERN "*"
          PATTERN "*.bat" EXCLUDE )
endif()
