set(LLVM_LINK_COMPONENTS
    Support
)

add_clang_tool(amdllvm
    amdllvm.cpp
)

list(APPEND CLANG_LINKS_TO_CREATE clang clang++ clang-cl clang-cpp clang-${CLANG_VERSION_MAJOR} lld)

foreach(link ${CLANG_LINKS_TO_CREATE})
    add_clang_symlink("amd${link}" amdllvm)
endforeach()
