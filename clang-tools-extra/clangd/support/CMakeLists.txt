# clangd/support contains low-level support libraries that do not depend
# on clang either programmatically or conceptually.

set(LLVM_LINK_COMPONENTS
  Support
  )

if(CLANG_BUILT_STANDALONE)
  # needed to get HAVE_CXX_ATOMICS64_WITHOUT_LIB defined
  include(CheckAtomic)
endif()

set(CLANGD_ATOMIC_LIB "")
if(NOT HAVE_CXX_ATOMICS_WITHOUT_LIB OR NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
  list(APPEND CLANGD_ATOMIC_LIB "atomic")
endif()

add_clang_library(clangdSupport
  Bracket.cpp
  Cancellation.cpp
  Context.cpp
  DirectiveTree.cpp
  FileCache.cpp
  Lex.cpp
  Logger.cpp
  Markup.cpp
  MemoryTree.cpp
  Path.cpp
  Shutdown.cpp
  ThreadCrashReporter.cpp
  Threading.cpp
  ThreadsafeFS.cpp
  Token.cpp
  Trace.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${CLANGD_ATOMIC_LIB}
  )
