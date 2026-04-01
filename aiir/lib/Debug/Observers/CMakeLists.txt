add_aiir_library(AIIRObservers
  ActionLogging.cpp
  ActionProfiler.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Debug/Observers

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
  AIIRSupport
  AIIRIR
)
