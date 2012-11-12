// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// FIXME: [temp.deduct.conv]p2 bullets 1 and 2 can't actually happen without
// references?
// struct ConvertibleToArray {
//   //  template<typename T, unsigned N>
//   //  operator T(()[]) const;

// private:
//   typedef int array[17];

//   operator array() const;
// };

// void test_array(ConvertibleToArray cta) {
//   int *ip = cta;
//   ip = cta;
//   const float *cfp = cta;
// }

// bullet 2
// struct ConvertibleToFunction {
//   template<typename T, typename A1, typename A2>
//   operator T(A1, A2) const ()  { };
// };

// bullet 3
struct ConvertibleToCVQuals {
  template<typename T>
  operator T* const() const;
};

void test_cvqual_conv(ConvertibleToCVQuals ctcv) {
  int *ip = ctcv;
  const int *icp = ctcv;
}
