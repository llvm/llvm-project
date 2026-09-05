#ifndef WHERE_OBJECT_QUALIFIERS_DIAG_H
#define WHERE_OBJECT_QUALIFIERS_DIAG_H

void invalidGlobal();

struct ObjectDiagBuilder {
  void objectOnly();
  void emptyObject();
  void duplicate();
  void acceptedDifferentRef() &;
  void acceptedDifferentRef() &&;
};

#endif // WHERE_OBJECT_QUALIFIERS_DIAG_H
