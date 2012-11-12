clang_tablegen(Options.inc -gen-opt-parser-defs
  SOURCE Options.td
  TARGET ClangDriverOptions)

clang_tablegen(CC1AsOptions.inc -gen-opt-parser-defs
  SOURCE CC1AsOptions.td
  TARGET ClangCC1AsOptions)
