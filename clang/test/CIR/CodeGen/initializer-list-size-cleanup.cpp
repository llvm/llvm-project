// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

namespace std {
template <typename T> class initializer_list {
  const T *data;
  __SIZE_TYPE__ len;

public:
  initializer_list();
};
} // namespace std

struct Elem {
  Elem(int);
  ~Elem();
};

struct Container {
  Container(std::initializer_list<Elem>);
  ~Container();
};

void build_container() {
  Container c = {1, 2, 3};
}

// CIR-LABEL: cir.func {{.*}}@_Z15build_containerv
// CIR:         cir.cleanup.scope {
// CIR:           %[[LEN_CONST:.*]] = cir.const #cir.int<3> : !u64i
// CIR:           %[[LEN_PTR:.*]] = cir.get_member {{.*}}[1] {name = "len"}
// CIR:           cir.store {{.*}} %[[LEN_CONST]], %[[LEN_PTR]]
