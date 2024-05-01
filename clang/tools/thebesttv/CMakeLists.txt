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

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

# add debug
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# enable exceptions
target_compile_options(thebesttv PRIVATE -fexceptions)

target_link_libraries(thebesttv
    PRIVATE
    spdlog::spdlog
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
