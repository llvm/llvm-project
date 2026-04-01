set(LLVM_LINK_COMPONENTS Support)
set(LIBS AIIRSupport)

add_llvm_executable(aiir-src-sharder aiir-src-sharder.cpp)

set_target_properties(aiir-src-sharder PROPERTIES FOLDER "AIIR/Tablegenning")
target_link_libraries(aiir-src-sharder PRIVATE ${LIBS})

set(AIIR_SRC_SHARDER_TABLEGEN_EXE aiir-src-sharder PARENT_SCOPE)

aiir_check_all_link_libraries(aiir-src-sharder)
