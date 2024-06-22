set(LLVM_LINK_COMPONENTS Support)
add_clang_tool(thebesttv
    main.cpp
    utils.cpp
    GenICFG.cpp
    ICFG.cpp
    CompilationDatabase.cpp
    GenAST.cpp
    DumpPath.cpp
)

# add debug
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# enable exceptions
target_compile_options(thebesttv PRIVATE -fexceptions)

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
