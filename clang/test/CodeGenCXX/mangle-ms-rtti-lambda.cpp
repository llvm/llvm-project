// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -fms-compatibility -fms-extensions -emit-llvm -o - %s | FileCheck --implicit-check-not='?$make_layer' %s

namespace std {
class type_info {
public:
  const char *name() const;
};
}

using size_t = decltype(sizeof(0));

template <typename... Ts> struct pack {};

template <typename A, typename B> struct pair_like {
  using first = A;
  using second = B;
};

template <typename A, typename B, typename C, typename D> struct quad_like {
  using a = A;
  using b = B;
  using c = C;
  using d = D;
};

template <typename T> struct holder {
  T *value;
};

template <typename T> auto make_layer() {
  auto local_lambda = [](const T *, size_t) noexcept { return sizeof(T); };
  using L = decltype(local_lambda);
  using PairTL = pair_like<T, L>;
  using PairLT = pair_like<L, T>;
  using Quad = quad_like<PairTL, PairLT, holder<T>, holder<L>>;
  return pack<T, L, PairTL, PairLT, Quad, holder<Quad>>{};
}

struct seed {
  int value;
};

using t01 = decltype(make_layer<seed>());
using t02 = decltype(make_layer<t01>());
using t03 = decltype(make_layer<t02>());

extern "C" const std::type_info *repro_type_infos[] = {
    &typeid(t01),
    &typeid(t02),
    &typeid(t03),
};

extern "C" const char *repro_last_name() {
  return typeid(t03).name();
}

// CHECK-DAG: global %rtti.TypeDescriptor{{[0-9]+}} { ptr @"??_7type_info@@6B@", ptr null, [{{[0-9]+}} x i8] c".?AU?$pack@Useed@@V<lambda_1>@@U?$pair_like@Useed@@V<lambda_1>@@@@
// CHECK-DAG: global %rtti.TypeDescriptor37 { ptr @"??_7type_info@@6B@", ptr null, [38 x i8] c".??@{{[0-9a-f]+}}@\00" }, comdat
