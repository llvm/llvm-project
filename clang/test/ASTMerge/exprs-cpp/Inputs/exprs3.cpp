// Integer literals
const char Ch1 = 'a';
const signed char Ch2 = 'b';
const unsigned char Ch3 = 'c';

const wchar_t Ch4 = L'd';
const signed wchar_t Ch5 = L'e';
const unsigned wchar_t Ch6 = L'f';

const short C1 = 12;
const unsigned short C2 = 13;

const int C3 = 12;
const unsigned int C4 = 13;

const long C5 = 22;
const unsigned long C6 = 23;

const long long C7 = 66;
const unsigned long long C8 = 67;


// String literals
const char str1[] = "ABCD";
const char str2[] = "ABCD" "0123";

const wchar_t wstr1[] = L"DEF";
const wchar_t wstr2[] = L"DEF" L"123";


// Boolean literals
const bool bval1 = true;
const bool bval2 = false;

// Floating Literals
const float F1 = 12.2F;
const double F2 = 1E4;
const long double F3 = 1.2E-3L;


// nullptr literal
const void *vptr = nullptr;


int glb_1[4] = { 10, 20, 30, 40 };

struct S1 {
  int a;
  int b[3];
};

struct S2 {
  int c;
  S1 d;
};

S2 glb_2 = { 22, .d.a = 44, .d.b[0] = 55, .d.b[1] = 66 };

void testNewThrowDelete() {
  throw;
  char *p = new char[10];
  delete[] p;
}

int testArrayElement(int *x, int n) {
  return x[n];
}

int testTernaryOp(int c, int x, int y) {
  return c ? x : y;
}

S1 &testConstCast(const S1 &x) {
  return const_cast<S1&>(x);
}

S1 &testStaticCast(S1 &x) {
  return static_cast<S1&>(x);
}

S1 &testReinterpretCast(S1 &x) {
  return reinterpret_cast<S1&>(x);
}

S1 &testDynamicCast(S1 &x) {
  return dynamic_cast<S1&>(x);
}

int testScalarInit(int x) {
  return int(x);
}

struct S {
  float f;
  double d;
};
struct T {
  int i;
  struct S s[10];
};

void testOffsetOf() {
  __builtin_offsetof(struct T, s[2].d);
}


int testDefaultArg(int a = 2*2) {
  return a;
}

int testDefaultArgExpr() {
  return testDefaultArg();
}

template <typename T> // T has TemplateTypeParmType
void testTemplateTypeParmType(int i);

void useTemplateType() {
  testTemplateTypeParmType<char>(4);
}

const bool ExpressionTrait = __is_lvalue_expr(1);
const unsigned ArrayRank = __array_rank(int[10][20]);
const unsigned ArrayExtent = __array_extent(int[10][20], 1);
