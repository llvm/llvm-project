// RUN: %check_clang_tidy %s cert-err34-c %t

typedef void *             FILE;

extern FILE *stdin;

extern int fscanf(FILE * stream, const char * format, ...);
extern int sscanf(const char * s, const char * format, ...);

extern double atof(const char *nptr);
extern int atoi(const char *nptr);
extern long int atol(const char *nptr);
extern long long int atoll(const char *nptr);

namespace std {
using ::FILE; using ::stdin;
using ::fscanf; using ::sscanf;
using ::atof; using ::atoi; using ::atol; using ::atoll;
}

void f1(const char *in) {
  int i;
  long long ll;

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'sscanf' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead [cert-err34-c]
  std::sscanf(in, "%d", &i);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'fscanf' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtoll' instead [cert-err34-c]
  std::fscanf(std::stdin, "%lld", &ll);
}

void f2(const char *in) {
  // CHECK-MESSAGES: :[[@LINE+1]]:11: warning: 'atoi' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead [cert-err34-c]
  int i = std::atoi(in); // to int
  // CHECK-MESSAGES: :[[@LINE+1]]:12: warning: 'atol' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead [cert-err34-c]
  long l = std::atol(in); // to long

  using namespace std;

  // CHECK-MESSAGES: :[[@LINE+1]]:18: warning: 'atoll' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtoll' instead [cert-err34-c]
  long long ll = atoll(in); // to long long
  // CHECK-MESSAGES: :[[@LINE+1]]:14: warning: 'atof' used to convert a string to a floating-point value, but function will not report conversion errors; consider using 'strtod' instead [cert-err34-c]
  double d = atof(in); // to double
}
