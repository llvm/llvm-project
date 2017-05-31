@import templates_top;

template<typename T> class Vector;

template<typename T> class Vector;

template<typename T> class List;
template<> class List<bool> {
public:
  void push_back(int);
};
namespace N {
  template<typename T> class Set;
}
namespace N {
  template<typename T> class Set {
  public:
    void insert(T);
  };
}

constexpr unsigned List<int>::*size_left = &List<int>::size;
List<int> list_left = { 0, 8 };
typedef List<int> ListInt_left;

template <typename T>
void pendingInstantiationEmit(T) {}
void triggerPendingInstantiation() {
  pendingInstantiationEmit(12);
  pendingInstantiationEmit(42.);
}

void redeclDefinitionEmit(){}

typedef Outer<int>::Inner OuterIntInner_left;

int defineListDoubleLeft() {
  List<double> ld;
  ld.push_back(0.0);
  return ld.size;
}

template<typename T> struct MergePatternDecl;

extern template struct ExplicitInstantiation<false, false>;
extern template struct ExplicitInstantiation<false, true>;
extern template struct ExplicitInstantiation<true, false>;
extern template struct ExplicitInstantiation<true, true>;

void useExplicitInstantiation() {
  ExplicitInstantiation<true, false>().f();
  ExplicitInstantiation<true, true>().f();
}

template<typename> struct DelayUpdates;
template<> struct DelayUpdates<int>;
template<typename T> struct DelayUpdates<T*>;
template<typename T> void testDelayUpdates(DelayUpdates<T> *p = 0) {}

void outOfLineInlineUseLeftF(void (OutOfLineInline<int>::*)() = &OutOfLineInline<int>::f);
void outOfLineInlineUseLeftG(void (OutOfLineInline<int>::*)() = &OutOfLineInline<int>::g);
void outOfLineInlineUseLeftH(void (OutOfLineInline<int>::*)() = &OutOfLineInline<int>::h);

namespace EmitDefaultedSpecialMembers {
  inline void f() {
    SmallString<256> SS;
  };
}

inline int *getStaticDataMemberLeft() {
  return WithUndefinedStaticDataMember<int[]>::undefined;
}
