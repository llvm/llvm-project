// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -verify 
// expected-no-diagnostics

#define SIZE 2

typedef struct {
  int noOfSymbols;
} Params;

static void create(const Params * const params, int fooList[]) {
  int tmpList[SIZE] = {0};
  for (int i = 0; i < params->noOfSymbols; i++)
    fooList[i] = tmpList[i];
}

int work(Params * const params) {
  int fooList[SIZE];
  create(params, fooList);
  int sum = 0;
  for (int i = 0; i < params->noOfSymbols; i++)
    sum += fooList[i]; // no-warning
  return sum;
}

static void create2(const Params * const * pparams, int fooList[]) {
  const Params * params = *pparams;
  int tmpList[SIZE] = {0};
  for (int i = 0; i < params->noOfSymbols; i++)
    fooList[i] = tmpList[i];
}

int work2(const Params * const params) {
  int fooList[SIZE];
  create2(&params, fooList);
  int sum = 0;
  for (int i = 0; i < params->noOfSymbols; i++)
    sum += fooList[i]; // no-warning
  return sum;
}

static void create3(Params * const * pparams, int fooList[]) {
  const Params * params = *pparams;
  int tmpList[SIZE] = {0};
  for (int i = 0; i < params->noOfSymbols; i++)
    fooList[i] = tmpList[i];
}

int work3(const Params * const params) {
  int fooList[SIZE];
  Params *const *ptr = (Params *const*)&params;
  create3(ptr, fooList);
  int sum = 0;
  for (int i = 0; i < params->noOfSymbols; i++)
    sum += fooList[i]; // no-warning
  return sum;
}

typedef Params ParamsTypedef;
typedef const ParamsTypedef *ConstParamsTypedef;

static void create4(ConstParamsTypedef const params, int fooList[]) {
  int tmpList[SIZE] = {0};
  for (int i = 0; i < params->noOfSymbols; i++)
    fooList[i] = tmpList[i];
}

int work4(Params * const params) {
  int fooList[SIZE];
  create4(params, fooList);
  int sum = 0;
  for (int i = 0; i < params->noOfSymbols; i++)
    sum += fooList[i]; // no-warning
  return sum;
}
