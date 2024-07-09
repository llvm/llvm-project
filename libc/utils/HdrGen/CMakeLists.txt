include(TableGen)

if (NOT LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS Support)
endif()

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

target_include_directories(libc-hdrgen PRIVATE ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})
target_link_libraries(libc-hdrgen PRIVATE LibcTableGenUtil)

add_subdirectory(PrototypeTestGen)
