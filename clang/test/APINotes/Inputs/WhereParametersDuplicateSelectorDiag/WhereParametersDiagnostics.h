#ifndef WHERE_PARAMETERS_DIAGNOSTICS_H
#define WHERE_PARAMETERS_DIAGNOSTICS_H

void duplicateGlobal(int);
void duplicateEmpty();
void allowedGlobal(int);
void allowedGlobal(double);

struct DiagnosticWidget {
  void duplicateMethod(int);
  void duplicateEmpty();
  void allowed(int);
  void allowed(double);
};

#endif // WHERE_PARAMETERS_DIAGNOSTICS_H
