#ifndef WHERE_PARAMETERS_DIAGNOSTICS_H
#define WHERE_PARAMETERS_DIAGNOSTICS_H

using DiagnosticAliasInt = int;

void unmatchedGlobal(float);
void diagnosticBroadGlobal(float);
void diagnosticMatchedGlobal(int);
void diagnosticAliasMatchedGlobal(DiagnosticAliasInt);

struct DiagnosticWidget {
  void unmatchedMethod(float);
  void diagnosticBroadMethod(float);
  void diagnosticMatchedMethod(int);
  void diagnosticAliasMatchedMethod(DiagnosticAliasInt);
};

#endif // WHERE_PARAMETERS_DIAGNOSTICS_H
