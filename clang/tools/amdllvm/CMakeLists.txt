set(LLVM_LINK_COMPONENTS
    Support
)

add_clang_tool(amdllvm
    amdllvm.cpp
)

option(CLANG_LINK_FLANG "Create flang install link to clang" ON)

list(APPEND CLANG_LINKS_TO_CREATE clang clang++ clang-cl clang-cpp clang-${CLANG_VERSION_MAJOR} lld)

if(CLANG_LINK_FLANG)
    list(APPEND CLANG_LINKS_TO_CREATE flang flang-new)
endif()

foreach(link ${CLANG_LINKS_TO_CREATE})
    add_clang_symlink("amd${link}" amdllvm)
endforeach()
