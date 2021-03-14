set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(IntrospectionTests
  IntrospectionTest.cpp
  )

clang_target_link_libraries(IntrospectionTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangTooling
  clangBasic
  clangSerialization
  clangFrontend
  )
target_compile_definitions(IntrospectionTests PRIVATE
  SKIP_INTROSPECTION_GENERATION=$<OR:$<CONFIG:Debug>,$<NOT:$<BOOL:${CLANG_TOOLING_BUILD_AST_INTROSPECTION}>>>
)
target_link_libraries(IntrospectionTests
  PRIVATE
  LLVMTestingSupport
)
