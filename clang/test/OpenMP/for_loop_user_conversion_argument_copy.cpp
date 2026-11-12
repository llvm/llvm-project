// RUN: %clang_cc1 -fsyntax-only -fopenmp -std=c++11 -verify %s
// expected-no-diagnostics

// Converting the initial value of a loop to the type of its iteration variable
// calls It(Start) as a user-defined conversion. Copying the Start into that
// constructor's parameter must not look for a user-defined conversion from it,
// because the copy constructor of Start can take it without one.

struct Start;

template <class T> struct IsConstStart {
  static const bool value = false;
};
template <> struct IsConstStart<const Start &> {
  static const bool value = true;
};

// Instantiated when looking for a conversion from T to Variant.
template <class T> struct ConvertibleToVariant {
  static_assert(!IsConstStart<T>::value,
                "looked for a user-defined conversion from Start");
  typedef void type;
};

struct Variant {
  template <class T, class = typename ConvertibleToVariant<T>::type>
  Variant(T &&);
};

struct Start {
  Start();
  Start(const Start &);
  Start(Variant);
};

struct It {
  It();
  It(const It &);
  It(Start);
  It &operator=(const It &);
  It &operator=(const Start &);
  It &operator+=(int);
  It &operator++();
  bool operator<(const It &) const;
};
int operator-(const It &, const It &);

void test(Start &start, const It &end) {
  It i;
#pragma omp parallel for
  for (i = start; i < end; ++i)
    ;
}
