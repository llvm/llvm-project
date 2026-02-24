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
  NumericLiteralCaseFixer.cpp
  NumericLiteralInfo.cpp
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
foreach(file IN LISTS files)
  add_custom_command(OUTPUT check_format_depend_${i}
    COMMAND clang-format ${file} | diff -u ${file} -
    VERBATIM
    COMMENT "Checking format of ${file}"
    )
  list(APPEND check_format_depends check_format_depend_${i})
  math(EXPR i ${i}+1)
endforeach()
add_custom_target(clang-format-check-format DEPENDS ${check_format_depends})

set(docs_tools_dir ${CLANG_SOURCE_DIR}/docs/tools)

set(format_style_depend ${CMAKE_CURRENT_BINARY_DIR}/format_style_depend)
set(dump_style dump_format_style.py)
set(style_options_rst ${CLANG_SOURCE_DIR}/docs/ClangFormatStyleOptions.rst)
add_custom_command(OUTPUT ${format_style_depend}
  COMMAND ${Python3_EXECUTABLE} ${dump_style} && touch ${format_style_depend}
  WORKING_DIRECTORY ${docs_tools_dir}
  VERBATIM
  COMMENT "Updating ${style_options_rst}"
  DEPENDS ${CLANG_SOURCE_DIR}/include/clang/Format/Format.h
          ${CLANG_SOURCE_DIR}/include/clang/Tooling/Inclusions/IncludeStyle.h
          ${style_options_rst}
          ${docs_tools_dir}/plurals.txt
          ${docs_tools_dir}/${dump_style}
  )
add_custom_target(clang-format-style DEPENDS ${format_style_depend})

set(format_help_depend ${CMAKE_CURRENT_BINARY_DIR}/format_help_depend)
set(dump_help dump_format_help.py)
set(clang_format_rst ${CLANG_SOURCE_DIR}/docs/ClangFormat.rst)
add_custom_command(OUTPUT ${format_help_depend}
  COMMAND ${Python3_EXECUTABLE} ${dump_help} -d ${CMAKE_BINARY_DIR}/bin &&
          touch ${format_help_depend}
  WORKING_DIRECTORY ${docs_tools_dir}
  VERBATIM
  COMMENT "Updating ${clang_format_rst}"
  DEPENDS clang-format
          ${clang_format_rst}
          ${docs_tools_dir}/${dump_help}
  )
add_custom_target(clang-format-help DEPENDS ${format_help_depend})
