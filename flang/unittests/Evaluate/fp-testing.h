#ifndef FORTRAN_TEST_EVALUATE_FP_TESTING_H_
#define FORTRAN_TEST_EVALUATE_FP_TESTING_H_

#include "flang/Common/target-rounding.h"
#include <fenv.h>

using Fortran::common::RealFlags;
using Fortran::common::Rounding;
using Fortran::common::RoundingMode;

class ScopedHostFloatingPointEnvironment {
public:
  ScopedHostFloatingPointEnvironment(bool treatSubnormalOperandsAsZero = false,
      bool flushSubnormalResultsToZero = false);
  ~ScopedHostFloatingPointEnvironment();
  void ClearFlags() const;
  static RealFlags CurrentFlags();
  static void SetRounding(Rounding rounding);

private:
  fenv_t originalFenv_;
#if __x86_64__
  unsigned int originalMxcsr;
#endif
};

#endif // FORTRAN_TEST_EVALUATE_FP_TESTING_H_
