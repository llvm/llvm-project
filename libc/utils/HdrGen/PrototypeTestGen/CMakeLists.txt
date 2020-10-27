add_tablegen(libc-prototype-testgen llvm-libc
  PrototypeTestGen.cpp
)
target_link_libraries(libc-prototype-testgen PRIVATE LibcTableGenUtil)
target_include_directories(libc-prototype-testgen PRIVATE ${LIBC_SOURCE_DIR})
