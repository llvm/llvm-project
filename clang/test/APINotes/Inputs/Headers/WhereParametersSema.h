#ifndef WHERE_PARAMETERS_SEMA_H
#define WHERE_PARAMETERS_SEMA_H

using AliasInt = int;
using AliasAliasInt = AliasInt;
using DeepAliasInt = AliasAliasInt;

void makeWidget(int);
void makeWidget(double);
void makeWidget();

void broadGlobal(int);
void broadGlobal(double);

void coexistGlobal(int);
void coexistGlobal(double);

void mismatchGlobal(float);
void aliasGlobal(AliasInt);
void aliasPrecedenceGlobal(AliasInt);
void multiAliasGlobal(DeepAliasInt);
void nullableGlobal(char * _Nonnull);
void rawIntGlobal(int);
void constValueGlobal(const int);

namespace SelectorNamespace {
void makeNamespaced(int);
void makeNamespaced(double);
}

struct SelectorWidget {
  void setValue(int);
  void setValue(double);
  void setValue();

  void broad(int);
  void broad(double);

  void coexist(int);
  void coexist(double);

  void defaults(int, double = 0);
  void defaults(int);

  static void configure(int);

  void mismatch(float);
  void alias(AliasInt);
  void aliasPrecedence(AliasInt);
  void multiAlias(DeepAliasInt);
  void nullable(char * _Nonnull);
  void rawInt(int);
  void constValue(const int);

  SelectorWidget operator+(int);
  SelectorWidget operator+(double);
};

#endif // WHERE_PARAMETERS_SEMA_H
