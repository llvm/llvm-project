// Verify ubsan vptr does not check down-casts on ignorelisted types.
// RUN: echo "type:_ZTI3Foo" > %t-type.ignorelist
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=vptr -fsanitize-recover=vptr -emit-llvm %s -o - | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=vptr -fsanitize-recover=vptr -fsanitize-ignorelist=%t-type.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=TYPE

class Bar {
public:
  virtual ~Bar() {}
};
class Foo : public Bar {};

Bar bar;

// DEFAULT: @_Z7checkmev
// TYPE: @_Z7checkmev
void checkme() {
// DEFAULT: call void @__ubsan_handle_dynamic_type_cache_miss({{.*}} ({{.*}}* @bar to
// TYPE-NOT: @__ubsan_handle_dynamic_type_cache_miss
  Foo* foo = static_cast<Foo*>(&bar); // down-casting
// DEFAULT: ret void
// TYPE: ret void
  return;
}
