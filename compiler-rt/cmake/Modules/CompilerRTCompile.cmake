# On Windows, CMAKE_*_FLAGS are built for MSVC but we use the GCC clang.exe,
# which uses completely different flags. Translate some common flag types, and
# drop the rest.
function(translate_msvc_cflags out_flags msvc_flags)
  # Insert an empty string in the list to simplify processing.
  set(msvc_flags ";${msvc_flags}")

  # Canonicalize /flag to -flag.
  string(REPLACE ";/" ";-" msvc_flags "${msvc_flags}")

  # Make space separated -D and -U flags into joined flags.
  string(REGEX REPLACE ";-\([DU]\);" ";-\\1" msvc_flags "${msvc_flags}")

  set(clang_flags "")
  foreach(flag ${msvc_flags})
    if ("${flag}" MATCHES "^-[DU]")
      # Pass through basic command line macro definitions (-DNDEBUG).
      list(APPEND clang_flags "${flag}")
    elseif ("${flag}" MATCHES "^-O[2x]")
      # Canonicalize normal optimization flags to -O2.
      list(APPEND clang_flags "-O2")
    endif()
  endforeach()
  set(${out_flags} "${clang_flags}" PARENT_SCOPE)
endfunction()

# Compile a source into an object file with COMPILER_RT_TEST_COMPILER using
# a provided compile flags and dependenices.
# clang_compile(<object> <source>
#               CFLAGS <list of compile flags>
#               DEPS <list of dependencies>)
macro(clang_compile object_file source)
  cmake_parse_arguments(SOURCE "" "" "CFLAGS;DEPS" ${ARGN})
  get_filename_component(source_rpath ${source} REALPATH)
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND SOURCE_DEPS clang compiler-rt-headers)
  endif()
  if (TARGET CompilerRTUnitTestCheckCxx)
    list(APPEND SOURCE_DEPS CompilerRTUnitTestCheckCxx)
  endif()
  string(REGEX MATCH "[.](cc|cpp)$" is_cxx ${source_rpath})
  if(is_cxx)
    string(REPLACE " " ";" global_flags "${CMAKE_CXX_FLAGS}")
  else()
    string(REPLACE " " ";" global_flags "${CMAKE_C_FLAGS}")
  endif()

  if (MSVC)
    translate_msvc_cflags(global_flags "${global_flags}")
  endif()

  if (APPLE)
    set(global_flags ${OSX_SYSROOT_FLAG} ${global_flags})
  endif()

  # Ignore unknown warnings. CMAKE_CXX_FLAGS may contain GCC-specific options
  # which are not supported by Clang.
  list(APPEND global_flags -Wno-unknown-warning-option)
  set(compile_flags ${global_flags} ${SOURCE_CFLAGS})
  add_custom_command(
    OUTPUT ${object_file}
    COMMAND ${COMPILER_RT_TEST_COMPILER} ${compile_flags} -c
            -o "${object_file}"
            ${source_rpath}
    MAIN_DEPENDENCY ${source}
    DEPENDS ${SOURCE_DEPS})
endmacro()

# On Darwin, there are no system-wide C++ headers and the just-built clang is
# therefore not able to compile C++ files unless they are copied/symlinked into
# ${LLVM_BINARY_DIR}/include/c++
# The just-built clang is used to build compiler-rt unit tests. Let's detect
# this before we try to build the tests and print out a suggestion how to fix
# it.
# On other platforms, this is currently not an issue.
macro(clang_compiler_add_cxx_check)
  if (APPLE)
    set(CMD
      "echo '#include <iostream>' | ${COMPILER_RT_TEST_COMPILER} ${OSX_SYSROOT_FLAG} -E -x c++ - > /dev/null"
      "if [ $? != 0 ] "
      "  then echo"
      "  echo 'Your just-built clang cannot find C++ headers, which are needed to build and run compiler-rt tests.'"
      "  echo 'You should copy or symlink your system C++ headers into ${LLVM_BINARY_DIR}/include/c++'"
      "  if [ -d $(dirname $(dirname $(xcrun -f clang)))/include/c++ ]"
      "    then echo 'e.g. with:'"
      "    echo '  cp -r' $(dirname $(dirname $(xcrun -f clang)))/include/c++ '${LLVM_BINARY_DIR}/include/'"
      "  elif [ -d $(dirname $(dirname $(xcrun -f clang)))/lib/c++ ]"
      "    then echo 'e.g. with:'"
      "    echo '  cp -r' $(dirname $(dirname $(xcrun -f clang)))/lib/c++ '${LLVM_BINARY_DIR}/include/'"
      "  fi"
      "  echo 'This can also be fixed by checking out the libcxx project from llvm.org and installing the headers'"
      "  echo 'into your build directory:'"
      "  echo '  cd ${LLVM_MAIN_SRC_DIR}/projects && svn co http://llvm.org/svn/llvm-project/libcxx/trunk libcxx'"
      "  echo '  cd ${LLVM_BINARY_DIR} && make -C ${LLVM_MAIN_SRC_DIR}/projects/libcxx installheaders HEADER_DIR=${LLVM_BINARY_DIR}/include'"
      "  echo"
      "  false"
      "fi"
      )
    add_custom_target(CompilerRTUnitTestCheckCxx
      COMMAND bash -c "${CMD}"
      COMMENT "Checking that just-built clang can find C++ headers..."
      VERBATIM)
    if (TARGET clang)
      ADD_DEPENDENCIES(CompilerRTUnitTestCheckCxx clang)
    endif()
  endif()
endmacro()
