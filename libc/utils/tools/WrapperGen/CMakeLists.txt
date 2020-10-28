set(LLVM_LINK_COMPONENTS Support)

add_tablegen(libc-wrappergen llvm-libc
  Main.cpp
)

target_include_directories(libc-wrappergen PRIVATE ${LIBC_SOURCE_DIR})
target_link_libraries(libc-wrappergen PRIVATE LibcTableGenUtil)
