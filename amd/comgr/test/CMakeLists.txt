# TODO: These targets depend on ALL to ensure they are up-to-date, but we
# should identify what they really need to depend on (it should be both the
# sources and the clang/llvm/lld being used).
set(CLANG_COMMON "${LLVM_TOOLS_BINARY_DIR}/clang" --target=amdgcn-amdhsa-opencl -mcpu=gfx803)
add_custom_target(bytes1.o ALL ${CLANG_COMMON}
  "${CMAKE_CURRENT_SOURCE_DIR}/reloc1.cl" -c -o
  "${CMAKE_CURRENT_BINARY_DIR}/reloc1.o"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reloc1.cl")
add_custom_target(bytes2.o ALL ${CLANG_COMMON}
  "${CMAKE_CURRENT_SOURCE_DIR}/reloc2.cl" -c -o
  "${CMAKE_CURRENT_BINARY_DIR}/reloc2.o"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reloc2.cl")
add_custom_target(shared.so ALL ${CLANG_COMMON}
  "${CMAKE_CURRENT_SOURCE_DIR}/shared.cl" -o
  "${CMAKE_CURRENT_BINARY_DIR}/shared.so"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shared.cl")
add_custom_target(include-b.h.pch ALL ${CLANG_COMMON}
  -Xclang -emit-pch
  "${CMAKE_CURRENT_SOURCE_DIR}/include-b.h" -o
  "${CMAKE_CURRENT_BINARY_DIR}/include-b.h.pch"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include-b.h"
  VERBATIM)
# TODO: Where is this from? We could also build this from source, but it is a
# standard x86-64 ELF SO, so it will probably not change.
configure_file("source1.cl" "source1.cl" COPYONLY)
configure_file("source2.cl" "source2.cl" COPYONLY)
configure_file("include-a.h" "include-a.h" COPYONLY)
configure_file("source1.s" "source1.s" COPYONLY)

if(UNIX)
set(CMAKE_C_FLAGS "-m64 -g -O0")
set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN/..")
endif()

add_definitions(-DTEST_OBJ_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")

# Test : Data Objects and Data Set
add_executable(data_test data_test.c)
target_link_libraries(data_test amd_comgr)
add_test(NAME data_test COMMAND data_test)

# Test : Disassembly of relocatable ELF
add_executable(disasm_llvm_reloc_test disasm_llvm_reloc_test.c)
target_link_libraries(disasm_llvm_reloc_test amd_comgr)
add_test(NAME disasm_llvm_reloc_test COMMAND disasm_llvm_reloc_test)

# Test : Disassembly of shared object ELF
add_executable(disasm_llvm_so_test disasm_llvm_so_test.c)
target_link_libraries(disasm_llvm_so_test amd_comgr)
add_test(NAME disasm_llvm_so_test COMMAND disasm_llvm_so_test)

# Test : TP-Metadata (Metadata properties for a target)
add_executable(metadata_tp_test metadata_tp_test.c)
target_link_libraries(metadata_tp_test amd_comgr)
add_test(NAME metadata_tp_test COMMAND metadata_tp_test)

# Test : Metadata YAML (Metadata of a code object)
add_executable(metadata_yaml_test metadata_yaml_test.c)
target_link_libraries(metadata_yaml_test amd_comgr)
add_test(NAME metadata_yaml_test COMMAND metadata_yaml_test)

# Test : Symbols
add_executable(symbols_test symbols_test.c)
target_link_libraries(symbols_test amd_comgr)
add_test(NAME symbols_test COMMAND symbols_test)

# Test : Iterate Symbols
add_executable(symbols_iterate_test symbols_iterate_test.c)
target_link_libraries(symbols_iterate_test amd_comgr)
add_test(NAME symbols_iterate_test COMMAND symbols_iterate_test)

# Test : Compile
add_executable(compile_test compile_test.c)
target_link_libraries(compile_test amd_comgr)
add_test(NAME compile_test COMMAND compile_test)

# Test : Compile Minimal
add_executable(compile_minimal_test compile_minimal_test.c)
target_link_libraries(compile_minimal_test amd_comgr)
add_test(NAME compile_minimal_test COMMAND compile_minimal_test)

# Test : Assemble
add_executable(assemble_test assemble_test.c)
target_link_libraries(assemble_test amd_comgr)
add_test(NAME assemble_test COMMAND assemble_test)

# Test : Linking
add_executable(link_test link_test.c)
target_link_libraries(link_test amd_comgr)
add_test(NAME link_test COMMAND link_test)

# Windows binaries have no equivalent to -rpath, so we must set their PATH to
# include the .lib/.dll directory.
if (NOT(UNIX))
  set_tests_properties(
    data_test
    disasm_llvm_reloc_test
    disasm_llvm_so_test
    metadata_tp_test
    metadata_yaml_test
    symbols_test
    symbols_iterate_test
    compile_test
    compile_minimal_test
    assemble_test
    link_test
    PROPERTIES ENVIRONMENT "PATH=$<TARGET_LINKER_FILE_DIR:amd_comgr>")
endif()
