template<typename> struct MergeSpecializations;
template<typename T> struct MergeSpecializations<T[]> {
  typedef int partially_specialized_in_c;
};
template<> struct MergeSpecializations<bool> {
  typedef int explicitly_specialized_in_c;
};

template<typename T> struct MergeTemplateDefinitions {
  static constexpr int f();
  static constexpr int g();
};
template<typename T> constexpr int MergeTemplateDefinitions<T>::g() { return 2; }

template<typename T1 = int>
struct MergeAnonUnionMember {
  MergeAnonUnionMember() { (void)values.t1; }
  union { int t1; } values;
};
inline MergeAnonUnionMember<> maum_c() { return {}; }

template<typename T> struct DontWalkPreviousDeclAfterMerging { struct Inner { typedef T type; }; };
typedef DontWalkPreviousDeclAfterMerging<char>::Inner dwpdam_typedef;

namespace TestInjectedClassName {
  template<typename T> struct X { X(); };
  typedef X<char[3]> C;
}
