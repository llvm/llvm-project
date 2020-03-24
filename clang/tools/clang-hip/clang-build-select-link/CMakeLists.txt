set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Object
  Support
  TransformUtils
  IPO
  )

add_clang_executable(clang-build-select-link ClangBuildSelectLink.cpp)
add_dependencies(clang clang-build-select-link)
install(TARGETS clang-build-select-link RUNTIME DESTINATION bin)

