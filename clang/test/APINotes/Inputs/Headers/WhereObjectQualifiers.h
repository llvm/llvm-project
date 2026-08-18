#ifndef WHERE_OBJECT_QUALIFIERS_H
#define WHERE_OBJECT_QUALIFIERS_H

struct ObjectBuffer {};

struct ObjectBuilder {
  void buildRef() &;
  void buildRef() &&;

  void buildConst() &;
  void buildConst() const &;

  void buildVolatile();
  void buildVolatile() volatile;

  void buildNone();
  void buildCombined() const volatile &;
  void buildConstRValue() const &&;
  void buildVolatileRValue() volatile &&;
  void buildObjectOnly(int) &;
  void buildObjectOnly(double) &;
  void buildObjectOnly(int) &&;
  void buildTwoObjectNotes() &;
  void buildObjectAndParameter(int) const;
  static void buildStatic();
  static void buildStaticObjectOnly();

  void consume(ObjectBuffer &);
  void consume(ObjectBuffer &&);
};

#if __cplusplus >= 202302L
struct ExplicitObjectBuilder {
  void buildExplicitConst(this const ExplicitObjectBuilder &);
  void buildExplicitLValue(this ExplicitObjectBuilder &);
};
#endif

#endif // WHERE_OBJECT_QUALIFIERS_H
