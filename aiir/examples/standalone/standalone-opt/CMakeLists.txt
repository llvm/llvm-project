set(LIBS
    AIIRArithDialect
    AIIROptLib
    AIIRRegisterAllDialects
    AIIRRegisterAllPasses
    AIIRStandalone
  )
add_llvm_executable(standalone-opt standalone-opt.cpp)

llvm_update_compile_flags(standalone-opt)
target_link_libraries(standalone-opt PRIVATE ${LIBS})

aiir_check_all_link_libraries(standalone-opt)
