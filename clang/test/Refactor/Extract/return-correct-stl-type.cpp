namespace std {

struct Traits {
  typedef char char_type;
};

template<typename TraitsType>
struct BasicString {
  typedef typename TraitsType::char_type value_type;
  value_type value() const;
  const value_type *data() const;
};

template<typename TraitsType>
BasicString<TraitsType>
operator+(const BasicString<TraitsType> &lhs,
          const BasicString<TraitsType> &rhs);
template<typename TraitsType>
BasicString<TraitsType>
operator+(const BasicString<TraitsType> &lhs,
          const char *rhs);

template<typename TraitsType>
struct BasicString;
typedef BasicString<Traits> String;

} // end namespace std

void returnCharTypeNotUselessValueType() {
// CHECK1: "static char extracted(const std::String &x) {\nreturn x.value();\n}\n\n" [[@LINE-1]]:1
// CHECK1: "static const char *extracted(const std::String &x) {\nreturn x.data();\n}\n\n" [[@LINE-2]]:1
  std::String x;
// return-char-begin: +1:9
  (void)x.value();
// return-char-end: +0:1
// return-data-begin: +1:9
  (void)x.data();
// return-data-end: +0:1
} // RUN: clang-refactor-test perform -action extract -selected=return-char -selected=return-data %s | FileCheck --check-prefix=CHECK1 %s

void operatorTypeInferral() {
// CHECK2: "static std::String extracted(const std::String &x) {\nreturn x + "hello";\n}\n\n" [[@LINE-1]]:1
// CHECK2: "static std::String extracted(const std::String &x) {\nreturn x + x;\n}\n\n" [[@LINE-2]]:1
  std::String x;
// infer-string1-begin: +1:10
  (void)(x + "hello");
// infer-string1-end: -1:21
// infer-string2-begin: +1:10
  (void)(x + x);
// infer-string2-end: -1:15
} // RUN: clang-refactor-test perform -action extract -selected=infer-string1 -selected=infer-string2 %s | FileCheck --check-prefix=CHECK2 %s
