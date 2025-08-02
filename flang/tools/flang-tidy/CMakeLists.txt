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

add_subdirectory(bugprone)
add_subdirectory(modernize)
add_subdirectory(openmp)
add_subdirectory(performance)
add_subdirectory(readability)

set(ALL_FLANG_TIDY_CHECKS
  flangTidyBugproneModule
  flangTidyModernizeModule
  flangTidyOpenMPModule
  flangTidyPerformanceModule
  flangTidyReadabilityModule
  )

set(ALL_FLANG_TIDY_CHECKS ${ALL_FLANG_TIDY_CHECKS} PARENT_SCOPE)

add_subdirectory(tool)
