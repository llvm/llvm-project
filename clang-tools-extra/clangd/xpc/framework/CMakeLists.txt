
set(SOURCES
    ClangdXPC.cpp)
add_clang_library(ClangdXPCLib SHARED
  ${SOURCES}
  DEPENDS
  clangd
)
create_clangd_xpc_framework(ClangdXPCLib "ClangdXPC")
