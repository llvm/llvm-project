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
  static void buildStatic();
  static void buildStaticObjectOnly();

  void consume(ObjectBuffer &);
  void consume(ObjectBuffer &&);
};

#endif // WHERE_OBJECT_QUALIFIERS_H
