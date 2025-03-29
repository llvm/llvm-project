add_clang_unittest(InstallAPITests
  HeaderFileTest.cpp
  FileListTest.cpp
  )

clang_target_link_libraries(InstallAPITests
  PRIVATE
  clangInstallAPI 
  )

target_link_libraries(InstallAPITests PRIVATE LLVMTestingSupport)
