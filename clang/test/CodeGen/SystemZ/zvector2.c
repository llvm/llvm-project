// RUN: %clang_cc1 -no-opaque-pointers -triple s390x-linux-gnu -target-cpu z14 -fzvector \
// RUN:  -O -emit-llvm -o - -W -Wall -Werror %s | FileCheck %s

volatile vector float ff, ff2;
volatile vector bool int bi;

void test_assign (void)
{
// CHECK-LABEL: test_assign
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: store volatile <4 x float> [[VAL]], <4 x float>* @ff
  ff = ff2;
}

void test_pos (void)
{
// CHECK-LABEL: test_pos
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: store volatile <4 x float> [[VAL]], <4 x float>* @ff
  ff = +ff2;
}

void test_neg (void)
{
// CHECK-LABEL: test_neg
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fneg <4 x float> [[VAL]]
  ff = -ff2;
}

void test_preinc (void)
{
// CHECK-LABEL: test_preinc
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fadd <4 x float> [[VAL]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  ++ff2;
}

void test_postinc (void)
{
// CHECK-LABEL: test_postinc
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fadd <4 x float> [[VAL]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  ff2++;
}

void test_predec (void)
{
// CHECK-LABEL: test_predec
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fadd <4 x float> [[VAL]], <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
  --ff2;
}

void test_postdec (void)
{
// CHECK-LABEL: test_postdec
// CHECK: [[VAL:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fadd <4 x float> [[VAL]], <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
  ff2--;
}

void test_add (void)
{
// CHECK-LABEL: test_add
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fadd <4 x float> [[VAL1]], [[VAL2]]
  ff = ff + ff2;
}

void test_add_assign (void)
{
// CHECK-LABEL: test_add_assign
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: %{{.*}} = fadd <4 x float> [[VAL2]], [[VAL1]]
  ff += ff2;
}

void test_sub (void)
{
// CHECK-LABEL: test_sub
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fsub <4 x float> [[VAL1]], [[VAL2]]
  ff = ff - ff2;
}

void test_sub_assign (void)
{
// CHECK-LABEL: test_sub_assign
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: %{{.*}} = fsub <4 x float> [[VAL1]], [[VAL2]]
  ff -= ff2;
}

void test_mul (void)
{
// CHECK-LABEL: test_mul
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fmul <4 x float> [[VAL1]], [[VAL2]]
  ff = ff * ff2;
}

void test_mul_assign (void)
{
// CHECK-LABEL: test_mul_assign
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: %{{.*}} = fmul <4 x float> [[VAL2]], [[VAL1]]
  ff *= ff2;
}

void test_div (void)
{
// CHECK-LABEL: test_div
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: %{{.*}} = fdiv <4 x float> [[VAL1]], [[VAL2]]
  ff = ff / ff2;
}

void test_div_assign (void)
{
// CHECK-LABEL: test_div_assign
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: %{{.*}} = fdiv <4 x float> [[VAL1]], [[VAL2]]
  ff /= ff2;
}

void test_cmpeq (void)
{
// CHECK-LABEL: test_cmpeq
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[CMP:%[^ ]+]] = fcmp oeq <4 x float> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ff == ff2;
}

void test_cmpne (void)
{
// CHECK-LABEL: test_cmpne
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[CMP:%[^ ]+]] = fcmp une <4 x float> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ff != ff2;
}

void test_cmpge (void)
{
// CHECK-LABEL: test_cmpge
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[CMP:%[^ ]+]] = fcmp oge <4 x float> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ff >= ff2;
}

void test_cmpgt (void)
{
// CHECK-LABEL: test_cmpgt
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[CMP:%[^ ]+]] = fcmp ogt <4 x float> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ff > ff2;
}

void test_cmple (void)
{
// CHECK-LABEL: test_cmple
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[CMP:%[^ ]+]] = fcmp ole <4 x float> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ff <= ff2;
}

void test_cmplt (void)
{
// CHECK-LABEL: test_cmplt
// CHECK: [[VAL1:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff
// CHECK: [[VAL2:%[^ ]+]] = load volatile <4 x float>, <4 x float>* @ff2
// CHECK: [[CMP:%[^ ]+]] = fcmp olt <4 x float> [[VAL1]], [[VAL2]]
// CHECK: %{{.*}} = sext <4 x i1> [[CMP]] to <4 x i32>
  bi = ff < ff2;
}

