set(LLVM_LINK_COMPONENTS Support)
add_clang_tool(thebesttv
    main.cpp
    utils.cpp
)

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
