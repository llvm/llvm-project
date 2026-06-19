#ifndef WHERE_PARAMETERS_NORMALIZATION_H
#define WHERE_PARAMETERS_NORMALIZATION_H

using NormalizationAliasInt = int;
using NormalizationAliasAliasInt = NormalizationAliasInt;
using NormalizationDeepAliasInt = NormalizationAliasAliasInt;
using NormalizationConstAliasInt = const int;

template <typename T, typename U> struct NormalizationBox {};

void normalizedEmpty();
void normalizedEmpty(int);

void normalizedDefaults(int, double = 0);
void normalizedDefaults(int);

void normalizedWhitespace(unsigned int);
void normalizedUnsigned(unsigned);
void normalizedTemplateSpacing(NormalizationBox<int, double>);
void normalizedPointerSpacing(int *);
void normalizedRValueReferenceSpacing(int &&);
void normalizedConstValue(const int);
void normalizedConstSpelling(int);
void normalizedConstSuffixSpelling(int);
void normalizedPointerConst(int *const);
void normalizedPointeeConst(const int *);
void normalizedPointeeConstMismatch(const int *);
void normalizedAlias(NormalizationAliasInt);
void normalizedDeepAlias(NormalizationDeepAliasInt);
void normalizedDeepAliasSource(NormalizationDeepAliasInt);
void normalizedIntermediateAliasMismatch(NormalizationDeepAliasInt);
void normalizedConstAlias(NormalizationConstAliasInt);
void normalizedNullable(char * _Nullable);
void normalizedRawInt(int);

struct NormalizationWidget {
  void empty();
  void empty(int);

  void defaults(int, double = 0);
  void defaults(int);

  static void configure(int);

  void pointerSpacing(int *);
  void pointeeConstMismatch(const int *);
  void deepAlias(NormalizationDeepAliasInt);
};

#endif // WHERE_PARAMETERS_NORMALIZATION_H
