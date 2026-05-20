// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -fclangir -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -emit-llvm %s -o - | FileCheck --check-prefix=OGCG %s
#include <typeinfo>

struct Item {
  const std::type_info &ti;
  const char *name;
  void *(*make)();
};

template<typename T> void *make_impl() { return new T; }
template<typename T> constexpr Item item(const char *name) {
  return { typeid(T), name, make_impl<T> };
}


struct A { virtual ~A(); };
struct B { virtual ~B(); };
struct C : virtual A, virtual B {};

extern constexpr Item items[] = {
  item<B>("B"), item<C>("C")
};

// LLVM: Function Attrs: noinline
// LLVM-LABEL: define linkonce_odr void @_ZTv0_n12_N1CD1Ev(
// LLVM-SAME: ptr noundef [[TMP0:%.*]]) #[[ATTR2:[0-9]+]] {
// LLVM-NEXT:    [[TMP2:%.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:    store ptr [[TMP0]], ptr [[TMP2]], align 8
// LLVM-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8
// LLVM-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP3]], align 1
// LLVM-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP4]], i64 -12
// LLVM-NEXT:    [[TMP6:%.*]] = load i32, ptr [[TMP5]], align 4
// LLVM-NEXT:    [[TMP7:%.*]] = sext i32 [[TMP6]] to i64
// LLVM-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[TMP3]], i64 [[TMP7]]
// LLVM-NEXT:    call void @_ZN1CD1Ev(ptr noundef nonnull align 8 dereferenceable(8) [[TMP8]]) #[[ATTR3]]
// LLVM-NEXT:    ret void
//
//
// LLVM: Function Attrs: noinline
// LLVM-LABEL: define linkonce_odr void @_ZTv0_n12_N1CD0Ev(
// LLVM-SAME: ptr noundef [[TMP0:%.*]]) #[[ATTR2]] {
// LLVM-NEXT:    [[TMP2:%.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:    store ptr [[TMP0]], ptr [[TMP2]], align 8
// LLVM-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8
// LLVM-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP3]], align 1
// LLVM-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[TMP4]], i64 -12
// LLVM-NEXT:    [[TMP6:%.*]] = load i32, ptr [[TMP5]], align 4
// LLVM-NEXT:    [[TMP7:%.*]] = sext i32 [[TMP6]] to i64
// LLVM-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[TMP3]], i64 [[TMP7]]
// LLVM-NEXT:    call void @_ZN1CD0Ev(ptr noundef nonnull align 8 dereferenceable(8) [[TMP8]]) #[[ATTR3]]
// LLVM-NEXT:    ret void

// OGCG: Function Attrs: noinline nounwind optnone
// OGCG-LABEL: define linkonce_odr void @_ZTv0_n12_N1CD1Ev(
// OGCG-SAME: ptr noundef [[THIS:%.*]]) unnamed_addr #[[ATTR2:[0-9]+]] comdat align 2 {
// OGCG-NEXT:  [[ENTRY:.*:]]
// OGCG-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// OGCG-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// OGCG-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[THIS1]], align 8
// OGCG-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 -12
// OGCG-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
// OGCG-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, ptr [[THIS1]], i32 [[TMP1]]
// OGCG-NEXT:    tail call void @_ZN1CD1Ev(ptr noundef nonnull align 8 dereferenceable(8) [[TMP2]]) #[[ATTR6]]
// OGCG-NEXT:    ret void
//
//
// OGCG: Function Attrs: noinline nounwind optnone
// OGCG-LABEL: define linkonce_odr void @_ZTv0_n12_N1CD0Ev(
// OGCG-SAME: ptr noundef [[THIS:%.*]]) unnamed_addr #[[ATTR2]] comdat align 2 {
// OGCG-NEXT:  [[ENTRY:.*:]]
// OGCG-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// OGCG-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// OGCG-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[THIS1]], align 8
// OGCG-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 -12
// OGCG-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
// OGCG-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, ptr [[THIS1]], i32 [[TMP1]]
// OGCG-NEXT:    tail call void @_ZN1CD0Ev(ptr noundef nonnull align 8 dereferenceable(8) [[TMP2]]) #[[ATTR6]]
// OGCG-NEXT:    ret void