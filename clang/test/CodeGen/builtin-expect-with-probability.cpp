// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -O1 -disable-llvm-passes | FileCheck %s --check-prefix=ALL --check-prefix=O1
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -O0 | FileCheck %s --check-prefix=ALL --check-prefix=O0
extern int global;

int expect_taken(int x) {
// ALL-LABEL: expect_taken
// O1:        call i64 @llvm.expect.with.probability.i64(i64 {{%.*}}, i64 1, double 9.000000e-01)
// O0-NOT:    @llvm.expect.with.probability

  if (__builtin_expect_with_probability(x == 100, 1, 0.9)) {
    return 0;
  }
  return x;
}

int expect_not_taken(int x) {
// ALL-LABEL: expect_not_taken
// O1:        call i64 @llvm.expect.with.probability.i64(i64 {{%.*}}, i64 0, double 9.000000e-01)
// O0-NOT:    @llvm.expect.with.probability

  if (__builtin_expect_with_probability(x == 100, 0, 0.9)) {
    return 0;
  }
  return x;
}

struct S {
  static constexpr int prob = 1;
};

template<typename T>
int expect_taken_template(int x) {
// ALL-LABEL: expect_taken_template
// O1:        call i64 @llvm.expect.with.probability.i64(i64 {{%.*}}, i64 1, double 1.000000e+00)
// O0-NOT:    @llvm.expect.with.probability

	if (__builtin_expect_with_probability (x == 100, 1, T::prob)) {
		return 0;
	}
	return x;
}

int f() {
  return expect_taken_template<S>(global);
}

int x;
extern "C" {
  int y(void);
}
void foo();

void expect_value_side_effects() {
// ALL-LABEL: expect_value_side_effects
// ALL: [[CALL:%.*]] = call i32 @y
// O1:  [[SEXT:%.*]] = sext i32 [[CALL]] to i64
// O1:  call i64 @llvm.expect.with.probability.i64(i64 {{%.*}}, i64 [[SEXT]], double 6.000000e-01)
// O0-NOT: @llvm.expect.with.probability

  if (__builtin_expect_with_probability(x, y(), 0.6))
    foo();
}

int switch_cond(int x) {
// ALL-LABEL: switch_cond
// O1:        call i64 @llvm.expect.with.probability.i64(i64 {{%.*}}, i64 1, double 8.000000e-01)
// O0-NOT:    @llvm.expect.with.probability

  switch (__builtin_expect_with_probability(x, 1, 0.8)) {
  case 0:
    x = x + 0;
  case 1:
    x = x + 1;
  case 2:
    x = x + 2;
  case 5:
    x = x + 5;
  default:
    x = x + 6;
  }
  return x;
}

constexpr double prob = 0.8;

int variable_expected(int stuff) {
// ALL-LABEL: variable_expected
// O1: call i64 @llvm.expect.with.probability.i64(i64 {{%.*}}, i64 {{%.*}}, double 8.000000e-01)
// O0-NOT: @llvm.expect.with.probability

  int res = 0;

  switch(__builtin_expect_with_probability(stuff, stuff, prob)) {
    case 0:
      res = 1;
      break;
    default:
      break;
  }
  return res;
}
