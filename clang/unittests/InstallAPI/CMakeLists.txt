add_clang_unittest(InstallAPITests
  HeaderFileTest.cpp
  FileListTest.cpp
  CLANG_LIBS
  clangInstallAPI 
  LINK_LIBS
  LLVMTestingSupport
  )
