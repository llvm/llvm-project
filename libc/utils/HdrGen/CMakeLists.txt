set(LLVM_LINK_COMPONENTS Support)

add_tablegen(libc-hdrgen llvm-libc
  Command.h
  Command.cpp
  Generator.cpp
  Generator.h
  IncludeFileCommand.cpp
  IncludeFileCommand.h
  Main.cpp
  PublicAPICommand.cpp
  PublicAPICommand.h
)

add_subdirectory(PrototypeTestGen)
