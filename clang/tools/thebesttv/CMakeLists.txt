set(LLVM_LINK_COMPONENTS Support)
add_clang_tool(thebesttv
    main.cpp
    utils.cpp
    GenICFG.cpp
    ICFG.cpp
)

# add debug
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

target_link_libraries(thebesttv
    PRIVATE
    clangAST
    clangBasic
    clangDriver
    clangFrontend
    clangRewriteFrontend
    clangSerialization
    clangStaticAnalyzerFrontend
    clangTooling
    clangToolingSyntax
)
