// RUN: %clang_cc1 -triple x86_64-unknown-linux -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OGCG
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-linux -std=c++20 -emit-llvm -o - %s | FileCheck %s %}

// CHECK-DAG: define {{.*}} @_ZeqRK6TargetS1_({{.*}}) [[NORMAL_ATTRS:#[0-9]+]]
// CHECK-DAG: define {{.*}} @_ZeqRK12StrictTargetS1_({{.*}}) [[STRICT_ATTRS:#[0-9]+]]
// CHECK-DAG: define {{.*}} @_ZN12AssignTargetaSERKS_({{.*}}) [[NORMAL_ATTRS]]
// CHECK-DAG: define {{.*}} @_ZN18StrictAssignTargetaSERKS_({{.*}}) [[STRICT_ATTRS]]

// Templates
// CHECK-DAG: define {{.*}} @_ZeqRK14TemplateTargetIdES2_({{.*}}) [[NORMAL_ATTRS]]
// CHECK-DAG: define {{.*}} @_ZeqRK14TemplateTargetIfES2_({{.*}}) [[NORMAL_ATTRS]]
// CHECK-DAG: define {{.*}} @_ZeqRK20StrictTemplateTargetIdES2_({{.*}}) [[STRICT_ATTRS]]

// Templates defaulted outside the class
// CHECK-DAG: define {{.*}} @_ZNK15OutOfLineTargetIeEeqERKS0_({{.*}}) [[NORMAL_ATTRS]]
// CHECK-DAG: fcmp oeq x86_fp80
// CHECK-DAG: define {{.*}} @_ZNK21StrictOutOfLineTargetIeEeqERKS0_({{.*}}) [[STRICT_ATTRS]]
// OGCG-DAG: call i1 @llvm.experimental.constrained.fcmp.f80({{.*}}, metadata !"fpexcept.strict")

// --- NON-STRICT DECLARATIONS (at top of file, default FP is non-strict) ---

struct Target {
  double d;
  friend bool operator==(const Target&, const Target&) = default;
};

struct Member {
  double d;
  Member& operator=(const Member& Other) {
    d = Other.d + 1.0;
    return *this;
  }
};

struct AssignTarget {
  Member m;
};

template <typename T>
struct TemplateTarget {
  T d;
  friend bool operator==(const TemplateTarget&, const TemplateTarget&) = default;
};

template <typename T>
struct OutOfLineTarget {
  T d;
  bool operator==(const OutOfLineTarget&) const;
};

template <typename T>
bool OutOfLineTarget<T>::operator==(const OutOfLineTarget&) const = default;

// Trigger instantiation of TemplateTarget<double>::operator== in non-strict context.
bool test_template_non_strict(TemplateTarget<double> a, TemplateTarget<double> b) {
  return a == b;
}


// --- STRICT CONTEXT (pragmas enabled) ---
#pragma STDC FENV_ACCESS ON

// Use-sites triggering synthesis of non-strict defaulted functions in strict context.

bool test_non_strict_cmp(Target a, Target b) {
  return a == b;
}

void test_non_strict_assign(AssignTarget& a, AssignTarget& b) {
  a = b;
}

// Trigger instantiation of TemplateTarget<float>::operator== in strict context.
// It should still be non-strict because the template was defined in non-strict context.
bool test_template_strict(TemplateTarget<float> a, TemplateTarget<float> b) {
  return a == b;
}

// Instantiate OutOfLineTarget<long double>::operator== in strict context.
// It should still be non-strict because it was defaulted in non-strict context.
template bool OutOfLineTarget<long double>::operator==(const OutOfLineTarget<long double>&) const;

// Strict declarations (must get strictfp)

struct StrictTarget {
  double d;
  friend bool operator==(const StrictTarget&, const StrictTarget&) = default;
};

bool test_strict_cmp(StrictTarget a, StrictTarget b) {
  return a == b;
}

struct StrictAssignTarget {
  Member m;
};

void test_strict_assign(StrictAssignTarget& a, StrictAssignTarget& b) {
  a = b;
}

template <typename T>
struct StrictTemplateTarget {
  T d;
  friend bool operator==(const StrictTemplateTarget&, const StrictTemplateTarget&) = default;
};


template <typename T>
struct StrictOutOfLineTarget {
  T d;
  bool operator==(const StrictOutOfLineTarget&) const;
};

template <typename T>
bool StrictOutOfLineTarget<T>::operator==(const StrictOutOfLineTarget&) const = default;


// --- NON-STRICT CONTEXT AGAIN ---
#pragma STDC FENV_ACCESS OFF

// Trigger instantiation of StrictTemplateTarget<double>::operator== in non-strict context.
// It should be strict because the template was defined in strict context.
bool test_strict_template_non_strict(StrictTemplateTarget<double> a, StrictTemplateTarget<double> b) {
  return a == b;
}

// Trigger instantiation of StrictOutOfLineTarget<long double>::operator== in non-strict context.
// It should be strict because it was defaulted in strict context.
bool test_strict_out_of_line_non_strict(StrictOutOfLineTarget<long double> a, StrictOutOfLineTarget<long double> b) {
  return a == b;
}


// CHECK-DAG: attributes [[STRICT_ATTRS]] = { {{.*}}strictfp{{.*}} }
// CHECK-DAG: attributes [[NORMAL_ATTRS]] = { {{.*}} }
// CHECK-NOT: attributes [[NORMAL_ATTRS]] = { {{.*}}strictfp
