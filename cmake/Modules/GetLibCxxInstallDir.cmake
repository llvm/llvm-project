
# get libcxx install directory
#
# usage:
# get_libcxx_install_dir(library_dir_var)
#
# determine the path to install libc++ (usually LIBCXX_INSTALL_INCLUDE_TARGET_DIR)
#
# This logic was broken out of libcxx/CMakeLists.txt as both libcxx and clang's
# driver.cpp need to know at this
#

function(get_libcxx_install_dir lib_dir_var)
  set(LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")

  if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
    set(TARGET_SUBDIR ${LLVM_DEFAULT_TARGET_TRIPLE})
    if(LIBDIR_SUBDIR)
      string(APPEND TARGET_SUBDIR /${LIBDIR_SUBDIR})
    endif()
    set(ret_dir lib${LLVM_LIBDIR_SUFFIX}/${TARGET_SUBDIR})
  else()
    set(ret_dir lib${LIBDIR_SUFFIX})
  endif()

  set(${lib_dir_var} ${ret_dir} PARENT_SCOPE)
endfunction()

