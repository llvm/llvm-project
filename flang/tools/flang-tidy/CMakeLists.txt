set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
)

add_flang_library(flangTidy STATIC
  FlangTidy.cpp
  FlangTidyCheck.cpp
  FlangTidyModule.cpp
  FlangTidyOptions.cpp
)

target_link_libraries(flangTidy
  PRIVATE
  FortranSupport
  FortranEvaluate
  FortranParser
  FortranSemantics
  flangFrontend
  flangFrontendTool
  )

set(ALL_FLANG_TIDY_CHECKS
  )

set(ALL_FLANG_TIDY_CHECKS ${ALL_FLANG_TIDY_CHECKS} PARENT_SCOPE)

add_subdirectory(tool)
