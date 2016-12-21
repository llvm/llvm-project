#define SUPPRESS_NULLABILITY_WARNING(Type)                              \
  _Pragma("clang diagnostic push")                                      \
  _Pragma("clang diagnostic ignored \"-Wnullability-completeness\"")    \
  Type                                                                  \
  _Pragma("clang diagnostic pop")

void suppress1(SUPPRESS_NULLABILITY_WARNING(int *) ptr); // no warning

void shouldwarn5(int *ptr); //expected-warning{{missing a nullability type specifier}}

void trigger5(int * _Nonnull);

void suppress2(SUPPRESS_NULLABILITY_WARNING(int *) ptr); // no warning

