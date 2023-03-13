include(TableGen)

set(LLVM_LINK_COMPONENTS Support)

add_tablegen(libc-hdrgen LIBC
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

target_include_directories(libc-hdrgen PRIVATE ${LIBC_SOURCE_DIR})
target_link_libraries(libc-hdrgen PRIVATE LibcTableGenUtil)

add_subdirectory(PrototypeTestGen)
