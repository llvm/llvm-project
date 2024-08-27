// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header -fsyntax-only %s -verify

double indexi32(matrix<double,3,1> X, int       i) { return X[i][0]; }

double indexu32(matrix<double,3,1> X, uint      i) { return X[i][0]; }

double indexi16(matrix<double,3,1> X, int16_t   i) { return X[i][0]; }

double indexu16(matrix<double,3,1> X, uint16_t  i) { return X[i][0]; }

double indexi64(matrix<double,3,1> X, int64_t   i) { return X[i][0]; }

double indexu64(matrix<double,3,1> X, uint64_t  i) { return X[i][0]; }

double indexi32c(matrix<double,3,1> X, int      i) { return X[0][i]; }

double indexu32c(matrix<double,3,1> X, uint     i) { return X[0][i]; }

double indexi16c(matrix<double,3,1> X, int16_t  i) { return X[0][i]; }

double indexu16c(matrix<double,3,1> X, uint16_t i) { return X[0][i]; }

double indexi64c(matrix<double,3,1> X, int64_t  i) { return X[0][i]; }

double indexu64c(matrix<double,3,1> X, uint64_t i) { return X[0][i]; }

// expected-no-diagnostics
