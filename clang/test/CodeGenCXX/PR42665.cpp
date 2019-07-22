// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -std=c++17 -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -std=c++17 -O0 %s -o - | FileCheck %s

// Minimal reproducer for PR42665.

struct Foo {
  Foo() = default;
  virtual ~Foo() = default;
};

template <typename Deleter>
struct Pair {
  Foo first;
  Deleter second;
};

template <typename Deleter>
Pair(Foo, Deleter) -> Pair<Deleter>;

template <typename T>
void deleter(T& t) { t.~T(); }

auto make_pair() {
  return Pair{ Foo(), deleter<Foo> };
}

void foobar() {
  auto p = make_pair();
  auto& f = p.first;
  auto& d = p.second;
  d(f); // Invoke virtual destructor of Foo through d.
} // p's destructor is invoked.

// Regexes are used to handle both kind of mangling.
//
// CHECK-LABEL: define linkonce_odr{{( dso_local)?}} void @{{.*deleter.*Foo.*}}(%struct.Foo* dereferenceable({{[0-9]+}})
// CHECK-SAME: [[T:%.*]])
// CHECK: [[T_ADDR:%.*]] = alloca %struct.Foo*
// CHECK: store %struct.Foo* [[T]], %struct.Foo** [[T_ADDR]]
// CHECK: [[R0:%.*]] = load %struct.Foo*, %struct.Foo** [[T_ADDR]]
// CHECK: [[R1:%.*]] = bitcast %struct.Foo* [[R0]] to [[TYPE:.*struct\.Foo.*]]***
// CHECK: [[VTABLE:%.*]] = load [[TYPE]]**, [[TYPE]]*** [[R1]]
// CHECK: [[VFUN:%.*]] = getelementptr inbounds [[TYPE]]*, [[TYPE]]** [[VTABLE]], i64 0
// CHECK: [[DTOR:%.*]] = load [[TYPE]]*, [[TYPE]]** [[VFUN]]
// CHECK: call {{(void|i8\*)}} [[DTOR]](%struct.Foo* [[R0]]
//
// CHECK-LABEL: define{{( dso_local)?}} void @{{.*foobar.*}}()
// CHECK: [[P:%.*]] = alloca %struct.Pair
// CHECK: [[F:%.*]] = alloca %struct.Foo*
// CHECK: [[D:%.*]] = alloca [[TYPE:void \(%struct.Foo\*\)]]**
// CHECK: call void @{{.*make_pair.*}}(%struct.Pair* sret [[P]])
// CHECK: [[FIRST:%.*]] = getelementptr inbounds %struct.Pair, %struct.Pair* [[P]], i32 0, i32 0
// CHECK: store %struct.Foo* [[FIRST]], %struct.Foo** [[F]]
// CHECK: [[SECOND:%.*]] = getelementptr inbounds %struct.Pair, %struct.Pair* [[P]], i32 0, i32 1
// CHECK: store void (%struct.Foo*)** [[SECOND]], [[TYPE]]*** [[D]]
// CHECK: [[R0:%.*]] = load [[TYPE]]**, [[TYPE]]*** [[D]]
// CHECK: [[R1:%.*]] = load [[TYPE]]*, [[TYPE]]** [[R0]]
// CHECK: [[R2:%.*]] = load %struct.Foo*, %struct.Foo** [[F]]
// CHECK: call void [[R1]](%struct.Foo* dereferenceable({{[0-9]+}}) [[R2]])
// CHECK: call void @{{.*Pair.*Foo.*}}(%struct.Pair* [[P]])

