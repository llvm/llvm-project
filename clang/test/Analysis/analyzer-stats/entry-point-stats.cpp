// REQUIRES: asserts
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:    -analyzer-config dump-entry-point-stats-to-csv="%t.csv" \
// RUN:    -verify %s
// RUN: %csv2json "%t.csv" | FileCheck --check-prefix=CHECK %s
//
// CHECK:      {
// CHECK-NEXT:   "c:@F@fib#i#": {
// CHECK-NEXT:     "File": "{{.*}}entry-point-stats.cpp",
// CHECK-NEXT:     "DebugName": "fib(unsigned int)",
// CHECK-NEXT:     "PathRunningTime": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxBugClassSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxCFGSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxQueueSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxReachableSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxTimeSpentSolvingZ3Queries": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxValidBugClassSize": "{{[0-9]+}}",
// CHECK-NEXT:     "NumBlocks": "{{[0-9]+}}",
// CHECK-NEXT:     "NumBlocksUnreachable": "{{[0-9]+}}",
// CHECK-NEXT:     "NumCTUSteps": "{{[0-9]+}}",
// CHECK-NEXT:     "NumFunctionTopLevel": "{{[0-9]+}}",
// CHECK-NEXT:     "NumInlinedCalls": "{{[0-9]+}}",
// CHECK-NEXT:     "NumMaxBlockCountReached": "{{[0-9]+}}",
// CHECK-NEXT:     "NumMaxBlockCountReachedInInlined": "{{[0-9]+}}",
// CHECK-NEXT:     "NumOfDynamicDispatchPathSplits": "{{[0-9]+}}",
// CHECK-NEXT:     "NumPathsExplored": "{{[0-9]+}}",
// CHECK-NEXT:     "NumReachedInlineCountMax": "{{[0-9]+}}",
// CHECK-NEXT:     "NumRemoveDeadBindings": "{{[0-9]+}}",
// CHECK-NEXT:     "NumSTUSteps": "{{[0-9]+}}",
// CHECK-NEXT:     "NumSteps": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportEQClassAborted": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportEQClassWasExhausted": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportPassesZ3": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportRefuted": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesRetriedWithoutInlining": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3ExhaustedRLimit": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3QueryAcceptsReport": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3QueryRejectEQClass": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3QueryRejectReport": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3SpendsTooMuchTimeOnASingleEQClass": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3TimedOut": "{{[0-9]+}}",
// CHECK-NEXT:     "NumZ3QueriesDone": "{{[0-9]+}}",
// CHECK-NEXT:     "TimeSpentSolvingZ3Queries": "{{[0-9]+}}"
// CHECK-NEXT:   },
// CHECK-NEXT:   "c:@F@main#I#**C#": {
// CHECK-NEXT:     "File": "{{.*}}entry-point-stats.cpp",
// CHECK-NEXT:     "DebugName": "main(int, char **)",
// CHECK-NEXT:     "PathRunningTime": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxBugClassSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxCFGSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxQueueSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxReachableSize": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxTimeSpentSolvingZ3Queries": "{{[0-9]+}}",
// CHECK-NEXT:     "MaxValidBugClassSize": "{{[0-9]+}}",
// CHECK-NEXT:     "NumBlocks": "{{[0-9]+}}",
// CHECK-NEXT:     "NumBlocksUnreachable": "{{[0-9]+}}",
// CHECK-NEXT:     "NumCTUSteps": "{{[0-9]+}}",
// CHECK-NEXT:     "NumFunctionTopLevel": "{{[0-9]+}}",
// CHECK-NEXT:     "NumInlinedCalls": "{{[0-9]+}}",
// CHECK-NEXT:     "NumMaxBlockCountReached": "{{[0-9]+}}",
// CHECK-NEXT:     "NumMaxBlockCountReachedInInlined": "{{[0-9]+}}",
// CHECK-NEXT:     "NumOfDynamicDispatchPathSplits": "{{[0-9]+}}",
// CHECK-NEXT:     "NumPathsExplored": "{{[0-9]+}}",
// CHECK-NEXT:     "NumReachedInlineCountMax": "{{[0-9]+}}",
// CHECK-NEXT:     "NumRemoveDeadBindings": "{{[0-9]+}}",
// CHECK-NEXT:     "NumSTUSteps": "{{[0-9]+}}",
// CHECK-NEXT:     "NumSteps": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportEQClassAborted": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportEQClassWasExhausted": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportPassesZ3": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesReportRefuted": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesRetriedWithoutInlining": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3ExhaustedRLimit": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3QueryAcceptsReport": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3QueryRejectEQClass": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3QueryRejectReport": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3SpendsTooMuchTimeOnASingleEQClass": "{{[0-9]+}}",
// CHECK-NEXT:     "NumTimesZ3TimedOut": "{{[0-9]+}}",
// CHECK-NEXT:     "NumZ3QueriesDone": "{{[0-9]+}}",
// CHECK-NEXT:     "TimeSpentSolvingZ3Queries": "{{[0-9]+}}"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NOT: non_entry_point

// expected-no-diagnostics
int non_entry_point(int end) {
  int sum = 0;
  for (int i = 0; i <= end; ++i) {
    sum += i;
  }
  return sum;
}

int fib(unsigned n) {
  if (n <= 1) {
    return 1;
  }
  return fib(n - 1) + fib(n - 2);
}

int main(int argc, char **argv) {
  int i = non_entry_point(argc);
  return i;
}
