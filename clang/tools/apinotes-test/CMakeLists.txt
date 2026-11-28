set(LLVM_LINK_COMPONENTS
  Option
  Support)

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(APINotesTestOptsTableGen)

add_clang_executable(apinotes-test
  APINotesTest.cpp

  DEPENDS
  APINotesTestOptsTableGen
  )
clang_target_link_libraries(apinotes-test PRIVATE
  clangAPINotes)
