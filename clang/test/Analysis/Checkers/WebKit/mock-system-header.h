#pragma clang system_header

template <typename T, typename CreateFunction>
void callMethod(CreateFunction createFunction) {
  createFunction()->method();
}

template <typename T, typename CreateFunction>
inline void localVar(CreateFunction createFunction) {
  T* obj = createFunction();
  obj->method();
}

template <typename T>
struct MemberVariable {
    T* obj { nullptr };
};
