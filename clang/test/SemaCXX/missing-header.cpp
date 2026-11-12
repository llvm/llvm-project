// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "not exist" // expected-error{{'not exist' file not found}}

class AnalysisDeclContext {};
static ControlFlowKind CheckFallThrough(AnalysisDeclContext &AC) {
  if (const GCCAsmStmt *AS = dyn_cast<GCCAsmStmt>(S)) {}
  bool NoReturnEdge = false;
}
