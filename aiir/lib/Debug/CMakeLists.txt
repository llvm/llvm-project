add_subdirectory(Observers)

add_aiir_library(AIIRDebug
  CLOptionsSetup.cpp
  DebugCounter.cpp
  ExecutionContext.cpp
  BreakpointManagers/FileLineColLocBreakpointManager.cpp
  DebuggerExecutionContextHook.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Debug

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
  AIIRIR
  AIIRSupport
  AIIRObservers
)

