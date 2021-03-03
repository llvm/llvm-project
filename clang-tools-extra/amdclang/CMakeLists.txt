if(UNIX AND LIBCXX_ENABLE_STATIC AND LIBCXXABI_ENABLE_STATIC)
  add_clang_tool(amdclang
    amdclang.cpp

    DEPENDS
    clang
    cxx_static
    cxxabi_static
    lld
  )

  # This tool has a dependency on C++17 filesystem library;
  # however, some Linux distros still don't have sufficient support for C++17.
  # Compile with clang and static link to libc++ to workaround this issue.
  set(CMAKE_CXX_COMPILER ${CMAKE_BINARY_DIR}/bin/clang++)
  set(CMAKE_CXX_LINKER ${CMAKE_CXX_COMPILER})
  target_compile_features(amdclang PRIVATE cxx_std_17)
  target_compile_options(amdclang PRIVATE  -nostdinc++)
  target_include_directories(amdclang PRIVATE ${CMAKE_BINARY_DIR}/include/c++/v1)
  target_link_libraries(amdclang PRIVATE c++abi pthread)
  target_link_options(amdclang PRIVATE -stdlib=libc++
                                      -L${CMAKE_BINARY_DIR}/lib
                                      -fuse-ld=lld
  )

  if(NOT CLANG_LINKS_TO_CREATE)
    set(CLANG_LINKS_TO_CREATE clang++ clang-cl clang-cpp flang)
  endif()
  foreach(link ${CLANG_LINKS_TO_CREATE})
    add_clang_symlink("amd${link}" amdclang)
  endforeach()
endif()