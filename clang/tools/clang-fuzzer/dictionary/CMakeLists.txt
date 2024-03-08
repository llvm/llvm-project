add_clang_executable(clang-fuzzer-dictionary
  dictionary.c
  )

set_target_properties(clang-fuzzer-dictionary
  PROPERTIES
  LINKER_LANGUAGE CXX
  )
