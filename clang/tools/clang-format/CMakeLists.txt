set(LLVM_LINK_COMPONENTS support)

add_clang_tool(clang-format
  ClangFormat.cpp
  )

set(CLANG_FORMAT_LIB_DEPS
  clangBasic
  clangFormat
  clangRewrite
  clangToolingCore
  )

clang_target_link_libraries(clang-format
  PRIVATE
  ${CLANG_FORMAT_LIB_DEPS}
  )

if( LLVM_LIB_FUZZING_ENGINE OR LLVM_USE_SANITIZE_COVERAGE )
  add_subdirectory(fuzzer)
endif()

install(PROGRAMS clang-format-bbedit.applescript
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-format)
install(PROGRAMS clang-format-diff.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-format)
install(PROGRAMS clang-format-sublime.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-format)
install(PROGRAMS clang-format.el
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-format)
install(PROGRAMS clang-format.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/clang"
  COMPONENT clang-format)
install(PROGRAMS git-clang-format
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clang-format)
