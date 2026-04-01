set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  InterfaceSupport.cpp
  StorageUniquer.cpp
  Timing.cpp
  ToolUtilities.cpp
  TypeID.cpp
)

add_aiir_library(AIIRSupport
  FileUtilities.cpp
  InterfaceSupport.cpp
  RawOstreamExtras.cpp
  StateStack.cpp
  StorageUniquer.cpp
  Timing.cpp
  ToolUtilities.cpp
  TypeID.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Support

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB})
