# Create an object library instead of a whole static archive.
add_library(clangNullRemoteCache OBJECT EXCLUDE_FROM_ALL
  NullClient.cpp
)
llvm_update_compile_flags(clangNullRemoteCache)
