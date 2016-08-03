// RUN: %check_clang_tidy %s mpi-type-mismatch %t -- -- -I %S/Inputs/mpi-type-mismatch

#include "mpimock.h"

void charNegativeTest() {
  int buf;
  MPI_Send(&buf, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'int' does not match the MPI datatype 'MPI_CHAR'

  short buf2;
  MPI_Send(&buf2, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'short' does not match the MPI datatype 'MPI_CHAR'

  long buf3;
  MPI_Send(&buf3, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'long' does not match the MPI datatype 'MPI_CHAR'

  int8_t buf4;
  MPI_Send(&buf4, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'int8_t' does not match the MPI datatype 'MPI_CHAR'

  uint16_t buf5;
  MPI_Send(&buf5, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint16_t' does not match the MPI datatype 'MPI_CHAR'

  long double _Complex buf6;
  MPI_Send(&buf6, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'long double _Complex' does not match the MPI datatype 'MPI_CHAR'

  std::complex<float> buf7;
  MPI_Send(&buf7, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'complex<float>' does not match the MPI datatype 'MPI_CHAR'
}

void intNegativeTest() {
  unsigned char buf;
  MPI_Send(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned char' does not match the MPI datatype 'MPI_INT'

  unsigned buf2;
  MPI_Send(&buf2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned int' does not match the MPI datatype 'MPI_INT'

  short buf3;
  MPI_Send(&buf3, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'short' does not match the MPI datatype 'MPI_INT'

  long buf4;
  MPI_Send(&buf4, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'long' does not match the MPI datatype 'MPI_INT'

  int8_t buf5;
  MPI_Send(&buf5, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'int8_t' does not match the MPI datatype 'MPI_INT'

  uint16_t buf6;
  MPI_Send(&buf6, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint16_t' does not match the MPI datatype 'MPI_INT'

  long double _Complex buf7;
  MPI_Send(&buf7, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'long double _Complex' does not match the MPI datatype 'MPI_INT'

  std::complex<float> buf8;
  MPI_Send(&buf8, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'complex<float>' does not match the MPI datatype 'MPI_INT'
}

void longNegativeTest() {
  char buf;
  MPI_Send(&buf, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'char' does not match the MPI datatype 'MPI_LONG'

  unsigned buf2;
  MPI_Send(&buf2, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned int' does not match the MPI datatype 'MPI_LONG'

  unsigned short buf3;
  MPI_Send(&buf3, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned short' does not match the MPI datatype 'MPI_LONG'

  unsigned long buf4;
  MPI_Send(&buf4, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned long' does not match the MPI datatype 'MPI_LONG'

  int8_t buf5;
  MPI_Send(&buf5, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'int8_t' does not match the MPI datatype 'MPI_LONG'

  uint16_t buf6;
  MPI_Send(&buf6, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint16_t' does not match the MPI datatype 'MPI_LONG'

  long double _Complex buf7;
  MPI_Send(&buf7, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'long double _Complex' does not match the MPI datatype 'MPI_LONG'

  std::complex<float> buf8;
  MPI_Send(&buf8, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'complex<float>' does not match the MPI datatype 'MPI_LONG'
}

void int8_tNegativeTest() {
  char buf;
  MPI_Send(&buf, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'char' does not match the MPI datatype 'MPI_INT8_T'

  unsigned buf2;
  MPI_Send(&buf2, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned int' does not match the MPI datatype 'MPI_INT8_T'

  short buf3;
  MPI_Send(&buf3, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'short' does not match the MPI datatype 'MPI_INT8_T'

  unsigned long buf4;
  MPI_Send(&buf4, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned long' does not match the MPI datatype 'MPI_INT8_T'

  uint8_t buf5;
  MPI_Send(&buf5, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint8_t' does not match the MPI datatype 'MPI_INT8_T'

  uint16_t buf6;
  MPI_Send(&buf6, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint16_t' does not match the MPI datatype 'MPI_INT8_T'

  long double _Complex buf7;
  MPI_Send(&buf7, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'long double _Complex' does not match the MPI datatype 'MPI_INT8_T'

  std::complex<float> buf8;
  MPI_Send(&buf8, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'complex<float>' does not match the MPI datatype 'MPI_INT8_T'
}

void complex_c_long_double_complexNegativeTest() {
  char buf;
  MPI_Send(&buf, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'char' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  unsigned buf2;
  MPI_Send(&buf2, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned int' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  short buf3;
  MPI_Send(&buf3, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'short' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  unsigned long buf4;
  MPI_Send(&buf4, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned long' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  uint8_t buf5;
  MPI_Send(&buf5, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint8_t' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  uint16_t buf6;
  MPI_Send(&buf6, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint16_t' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  double _Complex buf7;
  MPI_Send(&buf7, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'double _Complex' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'

  std::complex<float> buf8;
  MPI_Send(&buf8, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'complex<float>' does not match the MPI datatype 'MPI_C_LONG_DOUBLE_COMPLEX'
}

void complex_cxx_float_complexNegativeTest() {
  char buf;
  MPI_Send(&buf, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'char' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  unsigned buf2;
  MPI_Send(&buf2, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned int' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  short buf3;
  MPI_Send(&buf3, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'short' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  unsigned long buf4;
  MPI_Send(&buf4, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'unsigned long' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  uint8_t buf5;
  MPI_Send(&buf5, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint8_t' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  uint16_t buf6;
  MPI_Send(&buf6, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'uint16_t' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  double _Complex buf7;
  MPI_Send(&buf7, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'double _Complex' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'

  std::complex<double> buf8;
  MPI_Send(&buf8, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: buffer type 'complex<double>' does not match the MPI datatype 'MPI_CXX_FLOAT_COMPLEX'
}

void skippedTypesTests() {
  // typedefs, user defined MPI and nullptr types are skipped
  typedef char CHAR;
  CHAR buf;
  MPI_Send(&buf, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);

  typedef unsigned UNSIGNED;
  UNSIGNED buf2;
  MPI_Send(&buf2, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);

#define _MPI_LONG MPI_LONG
  int buf3;
  MPI_Send(&buf3, 1, _MPI_LONG, 0, 0, MPI_COMM_WORLD);

#define _MPI_CXX_FLOAT_COMPLEX MPI_CXX_FLOAT_COMPLEX
  short buf4;
  MPI_Send(&buf4, 1, _MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);

  MPI_Send(NULL, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
}

void positiveTests() {
  char buf;
  MPI_Send(&buf, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

  int buf2;
  MPI_Send(&buf2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

  long buf3;
  MPI_Send(&buf3, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);

  int8_t buf4;
  MPI_Send(&buf4, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);

  long double _Complex buf5;
  MPI_Send(&buf5, 1, MPI_C_LONG_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);

  std::complex<float> buf6;
  MPI_Send(&buf6, 1, MPI_CXX_FLOAT_COMPLEX, 0, 0, MPI_COMM_WORLD);

  uint8_t buf7;
  MPI_Send(&buf7, 1, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);

  uint16_t buf8;
  MPI_Send(&buf8, 1, MPI_UINT16_T, 0, 0, MPI_COMM_WORLD);

  // On some systems like PPC or ARM, 'char' is unsigned by default which is why
  // distinct signedness for the buffer and MPI type is tolerated.
  unsigned char buf9;
  MPI_Send(&buf9, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}
