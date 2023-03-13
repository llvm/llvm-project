// RUN: %check_clang_tidy %s readability-magic-numbers %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: readability-magic-numbers.IgnoredIntegerValues, value: "0;1;2;10;100;"}, \
// RUN:   {key: readability-magic-numbers.IgnoredFloatingPointValues, value: "3.14;2.71828;9.81;10000.0;101.0;0x1.2p3"}, \
// RUN:   {key: readability-magic-numbers.IgnoreBitFieldsWidths, value: false}, \
// RUN:   {key: readability-magic-numbers.IgnorePowersOf2IntegerValues, value: true}, \
// RUN:   {key: readability-magic-numbers.IgnoreTypeAliases, value: false}]}' \
// RUN: --

template <typename T, int V>
struct ValueBucket {
  T value[V];
};

int BadGlobalInt = 5;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 5 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

int IntSquarer(int param) {
  return param * param;
}

void BuggyFunction() {
  int BadLocalInt = 6;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 6 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  (void)IntSquarer(7);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 7 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  int LocalArray[15];
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 15 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  for (int ii = 0; ii < 22; ++ii)
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 22 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  {
    LocalArray[ii] = 3 * ii;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 3 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  }

  ValueBucket<int, 66> Bucket;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 66 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

class TwoIntContainer {
public:
  TwoIntContainer(int val) : anotherMember(val * val), yetAnotherMember(6), anotherConstant(val + val) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:73: warning: 6 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  int getValue() const;

private:
  int oneMember = 9;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 9 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

  int anotherMember;

  int yetAnotherMember;

  const int oneConstant = 2;

  const int anotherConstant;
};

int ValueArray[] = {3, 5, 0, 0, 0};
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 3 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
// CHECK-MESSAGES: :[[@LINE-2]]:24: warning: 5 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

float FloatPiVariable = 3.1415926535f;
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 3.1415926535f is a magic number; consider replacing it with a named constant [readability-magic-numbers]
double DoublePiVariable = 6.283185307;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 6.283185307 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

float SomeFloats[] = {0.5, 0x1.2p4};
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 0.5 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
// CHECK-MESSAGES: :[[@LINE-2]]:28: warning: 0x1.2p4 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

int getAnswer() {
  if (ValueArray[0] < ValueArray[1])
    return ValueArray[1];

  return -3; // FILENOTFOUND
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 3 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}

struct HardwareGateway {
   unsigned int Some: 5;
   // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 5 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
   unsigned int Bits: 7;
   // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 7 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
   unsigned int: 6;
   // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 6 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
   unsigned int Flag: 1; // no warning since this is suppressed by IgnoredIntegerValues rule
   unsigned int: 0;      // no warning since this is suppressed by IgnoredIntegerValues rule
   unsigned int Rest: 13;
   // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 13 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
   //
   unsigned int Another[3];
   // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 3 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
};

using NumberInTypeAlias = ValueBucket<int, 25>;
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: 25 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
typedef ValueBucket<char, 243> NumberInTypedef;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 243 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

/*
 * Clean code
 */

#define INT_MACRO 5

const int GoodGlobalIntConstant = 42;

constexpr int AlsoGoodGlobalIntConstant = 42;

int InitializedByMacro = INT_MACRO;

void SolidFunction() {
  const int GoodLocalIntConstant = 43;

  (void)IntSquarer(GoodLocalIntConstant);

  int LocalArray[INT_MACRO];

  ValueBucket<int, INT_MACRO> Bucket;
}

const int ConstValueArray[] = {7, 9};

const int ConstValueArray2D[2][2] = {{7, 9}, {13, 15}};

/*
 * no warnings for ignored values (specified in the configuration above)
 */
int GrandfatheredIntegerValues[] = {0, 1, 2, 10, 100, -1, -10, -100, 65536};

float GrandfatheredFloatValues[] = {3.14f, 3.14, 2.71828, 2.71828f, -1.01E2, 1E4, 0x1.2p3};

/*
 * no warnings for enums
 */
enum Smorgasbord {
  STARTER,
  ALPHA = 3,
  BETA = 1 << 5,
};

const float FloatPiConstant = 3.1415926535f;
const double DoublePiConstant = 6.283185307;

const float Angles[] = {45.0f, 90.0f, 135.0f};

double DoubleZeroIsAccepted = 0.0;
float FloatZeroIsAccepted = 0.0f;

namespace geometry {

template <typename T>
struct Point {
  T x;
  T y;

  explicit Point(T xval, T yval) noexcept : x{xval}, y{yval} {
  }
};

template <typename T>
struct Dimension {
  T x;
  T y;

  explicit Dimension(T xval, T yval) noexcept : x{xval}, y{yval} {
  }
};

template <typename T>
struct Rectangle {
  Point<T> origin;
  Dimension<T> size;
  T rotation; // angle of rotation around origin

  Rectangle(Point<T> origin_, Dimension<T> size_, T rotation_ = 0) noexcept : origin{origin_}, size{size_}, rotation{rotation_} {
  }

  bool contains(Point<T> point) const;
};

} // namespace geometry

const geometry::Rectangle<double> mandelbrotCanvas{geometry::Point<double>{-2.5, -1}, geometry::Dimension<double>{3.5, 2}};

// Simulate the macro magic in Google Test internal headers.
class AssertionHelper {
public:
  AssertionHelper(const char *Message, int LineNumber) : Message(Message), LineNumber(LineNumber) {}

private:
  const char *Message;
  int LineNumber;
};

#define ASSERTION_HELPER_AT(M, L) AssertionHelper(M, L)

#define ASSERTION_HELPER(M) ASSERTION_HELPER_AT(M, __LINE__)

void FunctionWithCompilerDefinedSymbol(void) {
  ASSERTION_HELPER("here and now");
}

// Prove that integer literals introduced by the compiler are accepted silently.
extern int ConsumeString(const char *Input);

const char *SomeStrings[] = {"alpha", "beta", "gamma"};

int TestCheckerOverreach() {
  int Total = 0;

  for (const auto *Str : SomeStrings) {
    Total += ConsumeString(Str);
  }

  return Total;
}

// Prove that using enumerations values don't produce warnings.
enum class Letter : unsigned {
    A, B, C, D, E, F, G, H, I, J
};

template<Letter x> struct holder  { Letter letter = x;  };
template<Letter x> struct wrapper { using h_type = holder<x>;  };

template struct wrapper<Letter::A>;
template struct wrapper<Letter::J>;
