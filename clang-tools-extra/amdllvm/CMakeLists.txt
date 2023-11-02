if(UNIX AND LIBCXX_ENABLE_STATIC AND LIBCXXABI_ENABLE_STATIC)
  add_clang_tool(amdllvm
    amdllvm.cpp

    DEPENDS
    clang
    runtimes
    lld
  )

  # This tool has a dependency on C++17 filesystem library;
  # however, some Linux distros still don't have sufficient support for C++17.
  # Compile with clang and static link to libc++ to workaround this issue.
  set(CMAKE_CXX_COMPILER ${CMAKE_BINARY_DIR}/bin/clang++)
  set(CMAKE_CXX_LINKER ${CMAKE_CXX_COMPILER})
  target_compile_features(amdllvm PRIVATE cxx_std_17)
  target_compile_options(amdllvm PRIVATE  -nostdinc++)
  target_include_directories(amdllvm PRIVATE ${CMAKE_BINARY_DIR}/include/c++/v1)
  target_include_directories(amdllvm PRIVATE ${CMAKE_BINARY_DIR}/include/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1)
  target_link_libraries(amdllvm PRIVATE c++abi pthread)
  target_link_options(amdllvm PRIVATE -stdlib=libc++
                                      -L${CMAKE_BINARY_DIR}/lib
                                      -fuse-ld=lld
  )
  option(CLANG_LINK_FLANG "Create flang install link to clang" ON)
  if(NOT CLANG_LINKS_TO_CREATE)
    if(CLANG_LINK_FLANG)
      set(CLANG_LINKS_TO_CREATE clang clang++ clang-cl clang-cpp clang-${CLANG_VERSION_MAJOR} flang
                              lld)
    else()
      set(CLANG_LINKS_TO_CREATE clang clang++ clang-cl clang-cpp clang-${CLANG_VERSION_MAJOR} lld)
    endif()
  endif()

  foreach(link ${CLANG_LINKS_TO_CREATE})
    add_clang_symlink("amd${link}" amdllvm)
  endforeach()
endif()
