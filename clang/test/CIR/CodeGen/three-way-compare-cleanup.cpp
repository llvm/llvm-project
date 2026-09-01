// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

namespace std {
struct strong_ordering {
  int v;
  constexpr strong_ordering(int v) : v(v) {}
  static const strong_ordering equal;
  static const strong_ordering less;
  static const strong_ordering greater;
};
inline constexpr strong_ordering strong_ordering::equal{0};
inline constexpr strong_ordering strong_ordering::less{-1};
inline constexpr strong_ordering strong_ordering::greater{1};
} // namespace std

struct Holder {
  int v;
  Holder(int x);
  ~Holder();
  operator int() const;
};

auto three_way_cmp_with_temp(int a) {
  return Holder(a).operator int() <=> 0;
}

// CIR-LABEL: cir.func {{.*}}three_way_cmp_with_temp
// CIR:         cir.call @_ZN6HolderC1Ei
// CIR:         cir.cleanup.scope {
// CIR:           %[[CONV:.*]] = cir.call @_ZNK6HoldercviEv
// CIR:           %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:           %[[LT:.*]] = cir.const #cir.int<-1> : !s32i
// CIR:           %[[EQ:.*]] = cir.const #cir.int<0> : !s32i
// CIR:           %[[GT:.*]] = cir.const #cir.int<1> : !s32i
// CIR:           %[[CMP_LT:.*]] = cir.cmp lt %[[CONV]], %[[ZERO]]
// CIR:           %[[SEL1:.*]] = cir.select if %[[CMP_LT]] then %[[LT]] else %[[GT]]
// CIR:           %[[CMP_EQ:.*]] = cir.cmp eq %[[CONV]], %[[ZERO]]
// CIR:           %[[RESULT:.*]] = cir.select if %[[CMP_EQ]] then %[[EQ]] else %[[SEL1]]
// CIR:           %[[FIELD:.*]] = cir.get_member {{.*}}[0] {name = "v"}
// CIR:           cir.store {{.*}} %[[RESULT]], %[[FIELD]]
// CIR:           cir.yield
// CIR:         } cleanup normal {
// CIR:           cir.call @_ZN6HolderD1Ev
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.return
