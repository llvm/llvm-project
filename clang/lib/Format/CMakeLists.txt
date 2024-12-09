set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangFormat
  AffectedRangeManager.cpp
  BreakableToken.cpp
  ContinuationIndenter.cpp
  DefinitionBlockSeparator.cpp
  Format.cpp
  FormatToken.cpp
  FormatTokenLexer.cpp
  IntegerLiteralSeparatorFixer.cpp
  MacroCallReconstructor.cpp
  MacroExpander.cpp
  MatchFilePath.cpp
  NamespaceEndCommentsFixer.cpp
  ObjCPropertyAttributeOrderFixer.cpp
  QualifierAlignmentFixer.cpp
  SortJavaScriptImports.cpp
  TokenAnalyzer.cpp
  TokenAnnotator.cpp
  UnwrappedLineFormatter.cpp
  UnwrappedLineParser.cpp
  UsingDeclarationsSorter.cpp
  WhitespaceManager.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangToolingCore
  clangToolingInclusions
  )

file(GLOB_RECURSE files
  *.cpp
  *.h
  ${CLANG_SOURCE_DIR}/include/clang/Format/*.h
  ${CLANG_SOURCE_DIR}/tools/clang-format/*.cpp
  ${CLANG_SOURCE_DIR}/unittests/Format/*.cpp
  ${CLANG_SOURCE_DIR}/unittests/Format/*.h
  )

set(check_format_depends)
set(i 0)
foreach (file IN LISTS files)
  add_custom_command(OUTPUT clang-format-check-format${i}
    COMMAND clang-format ${file} | diff -u ${file} -
    VERBATIM
    COMMENT "Checking format of ${file}..."
    )
  list(APPEND check_format_depends clang-format-check-format${i})

  math(EXPR i ${i}+1)
endforeach ()

add_custom_target(clang-format-check-format DEPENDS ${check_format_depends})

if(CLANG_INCLUDE_DOCS)
  set(style_options_depends ${CMAKE_CURRENT_BINARY_DIR}/dummy_output)
  set(docs_tools_dir ${CLANG_SOURCE_DIR}/docs/tools)
  add_custom_command(OUTPUT ${style_options_depends}
    COMMAND ${Python3_EXECUTABLE} dump_format_style.py &&
            touch ${style_options_depends}
    WORKING_DIRECTORY ${docs_tools_dir}
    DEPENDS ${CLANG_SOURCE_DIR}/include/clang/Format/Format.h
            ${CLANG_SOURCE_DIR}/include/clang/Tooling/Inclusions/IncludeStyle.h
            ${CLANG_SOURCE_DIR}/docs/ClangFormatStyleOptions.rst
            ${docs_tools_dir}/plurals.txt
            ${docs_tools_dir}/dump_format_style.py
    )
  add_custom_target(clang-format-style-options DEPENDS ${style_options_depends})
endif()
