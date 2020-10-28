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

include_directories(..)
add_clang_library(clangdSupport
  Cancellation.cpp
  Context.cpp
  Logger.cpp
  Markup.cpp
  MemoryTree.cpp
  Shutdown.cpp
  Threading.cpp
  ThreadsafeFS.cpp
  Trace.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${CLANGD_ATOMIC_LIB}
  )
