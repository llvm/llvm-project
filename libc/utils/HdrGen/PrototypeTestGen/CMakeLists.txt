include(TableGen)

set(LLVM_LINK_COMPONENTS Support)

add_tablegen(libc-prototype-testgen LLVM_LIBC
  PrototypeTestGen.cpp
)
target_link_libraries(libc-prototype-testgen
  PRIVATE
    LibcTableGenUtil
)
target_include_directories(libc-prototype-testgen
  PRIVATE
    ${LIBC_SOURCE_DIR}
    ${LLVM_INCLUDE_DIR}
    ${LLVM_MAIN_INCLUDE_DIR}
)
