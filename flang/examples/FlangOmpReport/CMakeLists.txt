add_llvm_library(flangOmpReport
  MODULE
  FlangOmpReport.cpp
  FlangOmpReportVisitor.cpp

  DEPENDS
  acc_gen
  flangFrontend
  omp_gen
)
