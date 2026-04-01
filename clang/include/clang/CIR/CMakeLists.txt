set(AIIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../aiir/include ) # --includedir
set(AIIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/aiir/include)
include_directories(${AIIR_INCLUDE_DIR})
include_directories(${AIIR_TABLEGEN_OUTPUT_DIR})

add_subdirectory(Dialect)
add_subdirectory(Interfaces)
