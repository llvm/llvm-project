set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Support)
# Reader and Writer define identically-named classes in anonymous namespaces
# (IdentifierTableInfo, ContextIDTableInfo, etc.). Manager and YAMLCompiler
# use "using namespace api_notes" which brings api_notes::Module into scope,
# conflicting with clang::Module.
set_source_files_properties(
  APINotesReader.cpp
  APINotesWriter.cpp
  APINotesYAMLCompiler.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangAPINotes
  APINotesManager.cpp
  APINotesReader.cpp
  APINotesTypes.cpp
  APINotesWriter.cpp
  APINotesYAMLCompiler.cpp
  LINK_LIBS
    clangBasic)
