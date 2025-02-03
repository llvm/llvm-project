// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +bf16 \
// RUN:   -disable-O0-optnone -Werror -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg \
// RUN: | opt -S -passes=inline \
// RUN: | FileCheck %s

extern "C" {

extern int normal_callee();

// == FUNCTION DECLARATIONS ==

int streaming_decl(void) __arm_streaming;
int streaming_compatible_decl(void) __arm_streaming_compatible;
int shared_za_decl(void) __arm_inout("za");
int preserves_za_decl(void) __arm_preserves("za");
int private_za_decl(void);
int agnostic_za_decl(void) __arm_agnostic("sme_za_state");

// == FUNCTION DEFINITIONS ==

// CHECK-LABEL: @streaming_caller()
// CHECK-SAME: #[[SM_ENABLED:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
 int streaming_caller() __arm_streaming {
  return normal_callee();
}

// CHECK: declare i32 @normal_callee() #[[NORMAL_DECL:[0-9]+]]


// CHECK-LABEL: @streaming_callee()
// CHECK-SAME: #[[SM_ENABLED]]
// CHECK: call i32 @streaming_decl() #[[SM_ENABLED_CALL:[0-9]+]]
//
 int streaming_callee() __arm_streaming {
  return streaming_decl();
}

// CHECK: declare i32 @streaming_decl() #[[SM_ENABLED_DECL:[0-9]+]]

// CHECK-LABEL: @streaming_compatible_caller()
// CHECK-SAME: #[[SM_COMPATIBLE:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
 int streaming_compatible_caller() __arm_streaming_compatible {
  return normal_callee();
}

// CHECK-LABEL: @streaming_compatible_callee()
// CHECK-SAME: #[[SM_COMPATIBLE]]
// CHECK: call i32 @streaming_compatible_decl() #[[SM_COMPATIBLE_CALL:[0-9]+]]
//
 int streaming_compatible_callee() __arm_streaming_compatible {
  return streaming_compatible_decl();
}

// CHECK: declare i32 @streaming_compatible_decl() #[[SM_COMPATIBLE_DECL:[0-9]+]]

// CHECK-LABEL: @locally_streaming_caller()
// CHECK-SAME: #[[SM_BODY:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
__arm_locally_streaming int locally_streaming_caller() {
  return normal_callee();
}

// CHECK-LABEL: @locally_streaming_callee()
// CHECK-SAME: #[[SM_BODY]]
// CHECK: call i32 @locally_streaming_caller() #[[SM_BODY_CALL:[0-9]+]]
//
__arm_locally_streaming int locally_streaming_callee() {
  return locally_streaming_caller();
}


// CHECK-LABEL: @shared_za_caller()
// CHECK-SAME: #[[ZA_SHARED:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
 int shared_za_caller() __arm_inout("za") {
  return normal_callee();
}

// CHECK-LABEL: @shared_za_callee()
// CHECK-SAME: #[[ZA_SHARED]]
// CHECK: call i32 @shared_za_decl() #[[ZA_SHARED_CALL:[0-9]+]]
//
 int shared_za_callee() __arm_inout("za") {
  return shared_za_decl();
}

// CHECK: declare i32 @shared_za_decl() #[[ZA_SHARED_DECL:[0-9]+]]


// CHECK-LABEL: @preserves_za_caller()
// CHECK-SAME: #[[ZA_PRESERVED:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
 int preserves_za_caller() __arm_preserves("za") {
  return normal_callee();
}

// CHECK-LABEL: @preserves_za_callee()
// CHECK-SAME: #[[ZA_PRESERVED]]
// CHECK: call i32 @preserves_za_decl() #[[ZA_PRESERVED_CALL:[0-9]+]]
//
 int preserves_za_callee() __arm_preserves("za") {
  return preserves_za_decl();
}

// CHECK: declare i32 @preserves_za_decl() #[[ZA_PRESERVED_DECL:[0-9]+]]


// CHECK-LABEL: @new_za_caller()
// CHECK-SAME: #[[ZA_NEW:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
__arm_new("za") int new_za_caller() {
  return normal_callee();
}

// CHECK-LABEL: @new_za_callee()
// CHECK-SAME: #[[ZA_NEW]]
// CHECK: call i32 @private_za_decl()
//
__arm_new("za") int new_za_callee() {
  return private_za_decl();
}

// CHECK: declare i32 @private_za_decl()

// CHECK-LABEL: @agnostic_za_caller()
// CHECK-SAME: #[[ZA_AGNOSTIC:[0-9]+]]
// CHECK: call i32 @normal_callee()
//
int agnostic_za_caller() __arm_agnostic("sme_za_state") {
  return normal_callee();
}

// CHECK-LABEL: @agnostic_za_callee()
// CHECK: call i32 @agnostic_za_decl() #[[ZA_AGNOSTIC_CALL:[0-9]+]]
//
int agnostic_za_callee() {
  return agnostic_za_decl();
}

// CHECK-LABEL: @agnostic_za_callee_live_za()
// CHECK: call i32 @agnostic_za_decl() #[[ZA_AGNOSTIC_CALL]]
//
int agnostic_za_callee_live_za() __arm_inout("za") {
  return agnostic_za_decl();
}

// Ensure that the attributes are correctly propagated to function types
// and also to callsites.
typedef void (*s_ptrty) (int, int) __arm_streaming;
typedef void (*sc_ptrty) (int, int) __arm_streaming_compatible;
typedef void (*sz_ptrty) (int, int) __arm_inout("za");
typedef void (*pz_ptrty) (int, int) __arm_preserves("za");

// CHECK-LABEL: @test_streaming_ptrty(
// CHECK-SAME: #[[NORMAL_DEF:[0-9]+]]
// CHECK: call void [[F:%.*]](i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) #[[SM_ENABLED_CALL]]
//
void test_streaming_ptrty(s_ptrty f, int x, int y) { return f(x, y); }
// CHECK-LABEL: @test_streaming_compatible_ptrty(
// CHECK-SAME: #[[NORMAL_DEF]]
// CHECK: call void [[F:%.*]](i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) #[[SM_COMPATIBLE_CALL]]
//
void test_streaming_compatible_ptrty(sc_ptrty f, int x, int y) { return f(x, y); }
// CHECK-LABEL: @test_shared_za(
// CHECK-SAME: #[[ZA_SHARED]]
// CHECK: call void [[F:%.*]](i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) #[[ZA_SHARED_CALL]]
//
void  test_shared_za(sz_ptrty f, int x, int y) __arm_inout("za") { return f(x, y); }
// CHECK-LABEL: @test_preserved_za(
// CHECK-SAME: #[[ZA_SHARED]]
// CHECK: call void [[F:%.*]](i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) #[[ZA_PRESERVED_CALL]]
//
void  test_preserved_za(pz_ptrty f, int x, int y) __arm_inout("za") { return f(x, y); }

// CHECK-LABEL: @test_indirect_streaming_ptrty(
// CHECK-SAME: #[[NORMAL_DEF:[0-9]+]]
// CHECK: call void [[F:%.*]](i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) #[[SM_ENABLED_CALL]]
//
typedef s_ptrty **indirect_s_ptrty;
void test_indirect_streaming_ptrty(indirect_s_ptrty fptr, int x, int y) { return (**fptr)(x, y); }
} // extern "C"

//
// Test that having the attribute in different places (on declaration and on type)
// both results in the attribute being applied to the type.
//

// CHECK-LABEL: @_Z24test_same_type_streamingv(
// CHECK:   call void @_Z10streaming1v() #[[SM_ENABLED_CALL]]
// CHECK:   call void @_Z10streaming2v() #[[SM_ENABLED_CALL]]
// CHECK:   call void @_Z20same_type_streaming1v() #[[SM_ENABLED_CALL]]
// CHECK:   call void @_Z20same_type_streaming2v() #[[SM_ENABLED_CALL]]
// CHECK:   ret void
// CHECK: }
// CHECK: declare void @_Z10streaming1v() #[[SM_ENABLED_DECL]]
// CHECK: declare void @_Z10streaming2v() #[[SM_ENABLED_DECL]]
// CHECK: declare void @_Z20same_type_streaming1v() #[[SM_ENABLED_DECL]]
// CHECK: declare void @_Z20same_type_streaming2v() #[[SM_ENABLED_DECL]]
void streaming1(void) __arm_streaming;
void streaming2() __arm_streaming;
decltype(streaming1) same_type_streaming1;
decltype(streaming2) same_type_streaming2;
void test_same_type_streaming() {
  streaming1();
  streaming2();
  same_type_streaming1();
  same_type_streaming2();
}

//
// Test overloading; the attribute is not required for overloaded types and
// does not apply if not specified.
//

// CHECK-LABEL: @_Z12overloadedfni(
// CHECK-SAME: #[[SM_ENABLED]]
int  overloadedfn(int x) __arm_streaming { return x; }
// CHECK-LABEL: @_Z12overloadedfnf(
// CHECK-SAME: #[[NORMAL_DEF]]
//
float overloadedfn(float x) { return x; }
// CHECK-LABEL: @_Z13test_overloadi(
// CHECK-SAME: #[[NORMAL_DEF]]
//
int test_overload(int x) { return overloadedfn(x); }
// CHECK-LABEL: @_Z13test_overloadf(
// CHECK-SAME: #[[NORMAL_DEF]]
//
float test_overload(float x) { return overloadedfn(x); }

// CHECK-LABEL: @_Z11test_lambdai(
// CHECK-SAME: #[[NORMAL_DEF]]
// CHECK: call noundef i32 @"_ZZ11test_lambdaiENK3$_0clEi"({{.*}}) #[[SM_ENABLED_CALL]]
//
// CHECK: @"_ZZ11test_lambdaiENK3$_0clEi"(
// CHECK-SAME: #[[SM_ENABLED]]
int test_lambda(int x) {
  auto F = [](int x)  __arm_streaming { return x; };
  return F(x);
}

// CHECK-LABEL: @_Z27test_template_instantiationv(
// CHECK-SAME: #[[NORMAL_DEF]]
// CHECK: call noundef i32 @_Z15template_functyIiET_S0_(i32 noundef 12) #[[SM_ENABLED_CALL]]
//
// CHECK: @_Z15template_functyIiET_S0_(
// CHECK-SAME: #[[SM_ENABLED]]
template <typename Ty>
Ty template_functy(Ty x)  __arm_streaming { return x; }
int test_template_instantiation() { return template_functy(12); }

//
// Test that arm_locally_streaming is inherited by future redeclarations,
// even when they don't specify the attribute.
//

// CHECK: define {{.*}} @_Z25locally_streaming_inheritv(
// CHECK-SAME: #[[SM_BODY]]
__arm_locally_streaming void locally_streaming_inherit();
void locally_streaming_inherit() {
  streaming_decl();
}

// Test that the attributes are propagated properly to calls
// when using a variadic template as indirection.
__attribute__((always_inline))
int call() { return 0; }

template <typename T, typename... Other>
__attribute__((always_inline))
int call(T f, Other... other) __arm_inout("za") {
    return f() + call(other...);
}

// CHECK: {{.*}} @_Z22test_variadic_templatev(
// CHECK:      call {{.*}} i32 @normal_callee() #[[NOUNWIND_CALL:[0-9]+]]
// CHECK-NEXT: call {{.*}} i32 @streaming_decl() #[[NOUNWIND_SM_ENABLED_CALL:[0-9]+]]
// CHECK-NEXT: call {{.*}} i32 @streaming_compatible_decl() #[[NOUNWIND_SM_COMPATIBLE_CALL:[0-9]+]]
// CHECK-NEXT: call {{.*}} i32 @shared_za_decl() #[[NOUNWIND_ZA_SHARED_CALL:[0-9]+]]
// CHECK-NEXT: call {{.*}} i32 @preserves_za_decl() #[[NOUNWIND_ZA_PRESERVED_CALL:[0-9]+]]
// CHECK-NEXT: add nsw
// CHECK-NEXT: add nsw
// CHECK-NEXT: add nsw
// CHECK-NEXT: add nsw
// CHECK-NEXT: ret
int test_variadic_template() __arm_inout("za") {
  return call(normal_callee,
              streaming_decl,
              streaming_compatible_decl,
              shared_za_decl,
              preserves_za_decl);
}

// CHECK: attributes #[[SM_ENABLED]] = { mustprogress noinline nounwind vscale_range(1,16) "aarch64_pstate_sm_enabled" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[NORMAL_DECL]] = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[SM_ENABLED_DECL]] = { "aarch64_pstate_sm_enabled" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[SM_COMPATIBLE]] = { mustprogress noinline nounwind "aarch64_pstate_sm_compatible" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[SM_COMPATIBLE_DECL]] = { "aarch64_pstate_sm_compatible" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[SM_BODY]] = { mustprogress noinline nounwind vscale_range(1,16) "aarch64_pstate_sm_body" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[ZA_SHARED]] = { mustprogress noinline nounwind "aarch64_inout_za" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[ZA_SHARED_DECL]] = { "aarch64_inout_za" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[ZA_PRESERVED]] = { mustprogress noinline nounwind "aarch64_preserves_za" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[ZA_PRESERVED_DECL]] = { "aarch64_preserves_za" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[ZA_NEW]] = { mustprogress noinline nounwind "aarch64_new_za" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[ZA_AGNOSTIC]] = { mustprogress noinline nounwind "aarch64_za_state_agnostic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[NORMAL_DEF]] = { mustprogress noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+bf16,+sme" }
// CHECK: attributes #[[SM_ENABLED_CALL]] = { "aarch64_pstate_sm_enabled" }
// CHECK: attributes #[[SM_COMPATIBLE_CALL]] = { "aarch64_pstate_sm_compatible" }
// CHECK: attributes #[[SM_BODY_CALL]] = { "aarch64_pstate_sm_body" }
// CHECK: attributes #[[ZA_SHARED_CALL]] = { "aarch64_inout_za" }
// CHECK: attributes #[[ZA_PRESERVED_CALL]] = { "aarch64_preserves_za" }
// CHECK: attributes #[[ZA_AGNOSTIC_CALL]] = { "aarch64_za_state_agnostic" }
// CHECK: attributes #[[NOUNWIND_CALL]] = { nounwind }
// CHECK: attributes #[[NOUNWIND_SM_ENABLED_CALL]] = { nounwind "aarch64_pstate_sm_enabled" }
// CHECK: attributes #[[NOUNWIND_SM_COMPATIBLE_CALL]] = { nounwind "aarch64_pstate_sm_compatible" }
// CHECK: attributes #[[NOUNWIND_ZA_SHARED_CALL]] = { nounwind "aarch64_inout_za" }
// CHECK: attributes #[[NOUNWIND_ZA_PRESERVED_CALL]] = { nounwind "aarch64_preserves_za" }

