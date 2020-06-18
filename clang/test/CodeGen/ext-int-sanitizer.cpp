// RUN: %clang_cc1 -triple x86_64-gnu-linux -fsanitize=array-bounds,enum,float-cast-overflow,integer-divide-by-zero,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,unsigned-integer-overflow,signed-integer-overflow,shift-base,shift-exponent -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s


// CHECK: define void @_Z6BoundsRA10_KiU7_ExtIntILi15EEi
void Bounds(const int (&Array)[10], _ExtInt(15) Index) {
  int I1 = Array[Index];
  // CHECK: %[[SEXT:.+]] = sext i15 %{{.+}} to i64
  // CHECK: %[[CMP:.+]] = icmp ult i64 %[[SEXT]], 10
  // CHECK: br i1 %[[CMP]]
  // CHECK: call void @__ubsan_handle_out_of_bounds
}

// CHECK: define void @_Z4Enumv
void Enum() {
  enum E1 { e1a = 0, e1b = 127 }
  e1;
  enum E2 { e2a = -1, e2b = 64 }
  e2;
  enum E3 { e3a = (1u << 31) - 1 }
  e3;

  _ExtInt(34) a = e1;
  // CHECK: %[[E1:.+]] = icmp ule i32 %{{.*}}, 127
  // CHECK: br i1 %[[E1]]
  // CHECK: call void @__ubsan_handle_load_invalid_value_abort
  _ExtInt(34) b = e2;
  // CHECK: %[[E2HI:.*]] = icmp sle i32 {{.*}}, 127
  // CHECK: %[[E2LO:.*]] = icmp sge i32 {{.*}}, -128
  // CHECK: %[[E2:.*]] = and i1 %[[E2HI]], %[[E2LO]]
  // CHECK: br i1 %[[E2]]
  // CHECK: call void @__ubsan_handle_load_invalid_value_abort
  _ExtInt(34) c = e3;
  // CHECK: %[[E3:.*]] = icmp ule i32 {{.*}}, 2147483647
  // CHECK: br i1 %[[E3]]
  // CHECK: call void @__ubsan_handle_load_invalid_value_abort
}

// CHECK: define void @_Z13FloatOverflowfd
void FloatOverflow(float f, double d) {
  _ExtInt(10) E = f;
  // CHECK: fcmp ogt float %{{.+}}, -5.130000e+02
  // CHECK: fcmp olt float %{{.+}}, 5.120000e+02
  _ExtInt(10) E2 = d;
  // CHECK: fcmp ogt double %{{.+}}, -5.130000e+02
  // CHECK: fcmp olt double %{{.+}}, 5.120000e+02
  _ExtInt(7) E3 = f;
  // CHECK: fcmp ogt float %{{.+}}, -6.500000e+01
  // CHECK: fcmp olt float %{{.+}}, 6.400000e+01
  _ExtInt(7) E4 = d;
  // CHECK: fcmp ogt double %{{.+}}, -6.500000e+01
  // CHECK: fcmp olt double %{{.+}}, 6.400000e+01
}

// CHECK: define void @_Z14UIntTruncationU7_ExtIntILi35EEjjy
void UIntTruncation(unsigned _ExtInt(35) E, unsigned int i, unsigned long long ll) {

  i = E;
  // CHECK: %[[LOADE:.+]] = load i35
  // CHECK: store i35 %[[LOADE]], i35* %[[EADDR:.+]]
  // CHECK: %[[LOADE2:.+]] = load i35, i35* %[[EADDR]]
  // CHECK: %[[CONV:.+]] = trunc i35 %[[LOADE2]] to i32
  // CHECK: %[[EXT:.+]] = zext i32 %[[CONV]] to i35
  // CHECK: %[[CHECK:.+]] = icmp eq i35 %[[EXT]], %[[LOADE2]]
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  E = ll;
  // CHECK: %[[LOADLL:.+]] = load i64
  // CHECK: %[[CONV:.+]] = trunc i64 %[[LOADLL]] to i35
  // CHECK: %[[EXT:.+]] = zext i35 %[[CONV]] to i64
  // CHECK: %[[CHECK:.+]] = icmp eq i64 %[[EXT]], %[[LOADLL]]
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort
}

// CHECK: define void @_Z13IntTruncationU7_ExtIntILi35EEiU7_ExtIntILi42EEjij
void IntTruncation(_ExtInt(35) E, unsigned _ExtInt(42) UE, int i, unsigned j) {

  j = E;
  // CHECK: %[[LOADE:.+]] = load i35
  // CHECK: store i35 %[[LOADE]], i35* %[[EADDR:.+]]
  // CHECK: %[[LOADE2:.+]] = load i35, i35* %[[EADDR]]
  // CHECK: %[[CONV:.+]] = trunc i35 %[[LOADE2]] to i32
  // CHECK: %[[EXT:.+]] = zext i32 %[[CONV]] to i35
  // CHECK: %[[CHECK:.+]] = icmp eq i35 %[[EXT]], %[[LOADE2]]
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  j = UE;
  // CHECK: %[[LOADUE:.+]] = load i42
  // CHECK: %[[CONV:.+]] = trunc i42 %[[LOADUE]] to i32
  // CHECK: %[[EXT:.+]] = zext i32 %[[CONV]] to i42
  // CHECK: %[[CHECK:.+]] = icmp eq i42 %[[EXT]], %[[LOADUE]]
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  // Note: also triggers sign change check.
  i = UE;
  // CHECK: %[[LOADUE:.+]] = load i42
  // CHECK: %[[CONV:.+]] = trunc i42 %[[LOADUE]] to i32
  // CHECK: %[[NEG:.+]] = icmp slt i32 %[[CONV]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 false, %[[NEG]]
  // CHECK: %[[EXT:.+]] = sext i32 %[[CONV]] to i42
  // CHECK: %[[CHECK:.+]] = icmp eq i42 %[[EXT]], %[[LOADUE]]
  // CHECK: %[[CHECKBOTH:.+]] = and i1 %[[SIGNCHECK]], %[[CHECK]]
  // CHECK: br i1 %[[CHECKBOTH]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  // Note: also triggers sign change check.
  E = UE;
  // CHECK: %[[LOADUE:.+]] = load i42
  // CHECK: %[[CONV:.+]] = trunc i42 %[[LOADUE]] to i35
  // CHECK: %[[NEG:.+]] = icmp slt i35 %[[CONV]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 false, %[[NEG]]
  // CHECK: %[[EXT:.+]] = sext i35 %[[CONV]] to i42
  // CHECK: %[[CHECK:.+]] = icmp eq i42 %[[EXT]], %[[LOADUE]]
  // CHECK: %[[CHECKBOTH:.+]] = and i1 %[[SIGNCHECK]], %[[CHECK]]
  // CHECK: br i1 %[[CHECKBOTH]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort
}

// CHECK: define void @_Z15SignChangeCheckU7_ExtIntILi39EEjU7_ExtIntILi39EEi
void SignChangeCheck(unsigned _ExtInt(39) UE, _ExtInt(39) E) {
  UE = E;
  // CHECK: %[[LOADEU:.+]] = load i39
  // CHECK: %[[LOADE:.+]] = load i39
  // CHECK: store i39 %[[LOADE]], i39* %[[EADDR:.+]]
  // CHECK: %[[LOADE2:.+]] = load i39, i39* %[[EADDR]]
  // CHECK: %[[NEG:.+]] = icmp slt i39 %[[LOADE2]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 %[[NEG]], false
  // CHECK: br i1 %[[SIGNCHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  E = UE;
  // CHECK: store i39 %[[LOADE2]], i39* %[[UEADDR:.+]]
  // CHECK: %[[LOADUE2:.+]] = load i39, i39* %[[UEADDR]]
  // CHECK: %[[NEG:.+]] = icmp slt i39 %[[LOADUE2]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 false, %[[NEG]]
  // CHECK: br i1 %[[SIGNCHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort
}

// CHECK: define void @_Z9DivByZeroU7_ExtIntILi11EEii
void DivByZero(_ExtInt(11) E, int i) {

  // Also triggers signed integer overflow.
  E / E;
  // CHECK: %[[EADDR:.+]] = alloca i11
  // CHECK: %[[E:.+]] = load i11, i11* %[[EADDR]]
  // CHECK: %[[E2:.+]] = load i11, i11* %[[EADDR]]
  // CHECK: %[[NEZERO:.+]] = icmp ne i11 %[[E2]], 0
  // CHECK: %[[NEMIN:.+]] = icmp ne i11 %[[E]], -1024
  // CHECK: %[[NENEG1:.+]] = icmp ne i11 %[[E2]], -1
  // CHECK: %[[OR:.+]] = or i1 %[[NEMIN]], %[[NENEG1]]
  // CHECK: %[[AND:.+]] = and i1 %[[NEZERO]], %[[OR]]
  // CHECK: br i1 %[[AND]]
  // CHECK: call void @__ubsan_handle_divrem_overflow_abort
}

// TODO:
//-fsanitize=shift: (shift-base, shift-exponent) Shift operators where the amount shifted is greater or equal to the promoted bit-width of the left hand side or less than zero, or where the left hand side is negative. For a signed left shift, also checks for signed overflow in C, and for unsigned overflow in C++. You can use -fsanitize=shift-base or -fsanitize=shift-exponent to check only left-hand side or right-hand side of shift operation, respectively.
// CHECK: define void @_Z6ShiftsU7_ExtIntILi9EEi
void Shifts(_ExtInt(9) E) {
  E >> E;
  // CHECK: %[[EADDR:.+]] = alloca i9
  // CHECK: %[[LHSE:.+]] = load i9, i9* %[[EADDR]]
  // CHECK: %[[RHSE:.+]] = load i9, i9* %[[EADDR]]
  // CHECK: %[[CMP:.+]] = icmp ule i9 %[[RHSE]], 8
  // CHECK: br i1 %[[CMP]]
  // CHECK: call void @__ubsan_handle_shift_out_of_bounds_abort

  E << E;
  // CHECK: %[[LHSE:.+]] = load i9, i9*
  // CHECK: %[[RHSE:.+]] = load i9, i9*
  // CHECK: %[[CMP:.+]] = icmp ule i9 %[[RHSE]], 8
  // CHECK: br i1 %[[CMP]]
  // CHECK: %[[ZEROS:.+]] = sub nuw nsw i9 8, %[[RHSE]]
  // CHECK: %[[CHECK:.+]] = lshr i9 %[[LHSE]], %[[ZEROS]]
  // CHECK: %[[SKIPSIGN:.+]] = lshr i9 %[[CHECK]], 1
  // CHECK: %[[CHECK:.+]] = icmp eq i9 %[[SKIPSIGN]]
  // CHECK: %[[PHI:.+]] = phi i1 [ true, %{{.+}} ], [ %[[CHECK]], %{{.+}} ]
  // CHECK: and i1 %[[CMP]], %[[PHI]]
  // CHECK: call void @__ubsan_handle_shift_out_of_bounds_abort
}

// CHECK: define void @_Z21SignedIntegerOverflowU7_ExtIntILi93EEiU7_ExtIntILi4EEiU7_ExtIntILi31EEi
void SignedIntegerOverflow(_ExtInt(93) BiggestE,
                           _ExtInt(4) SmallestE,
                           _ExtInt(31) JustRightE) {
  BiggestE + BiggestE;
  // CHECK: %[[LOADBIGGESTE2:.+]] = load i93
  // CHECK: store i93 %[[LOADBIGGESTE2]], i93* %[[BIGGESTEADDR:.+]]
  // CHECK: %[[LOAD1:.+]] = load i93, i93* %[[BIGGESTEADDR]]
  // CHECK: %[[LOAD2:.+]] = load i93, i93* %[[BIGGESTEADDR]]
  // CHECK: %[[OFCALL:.+]] = call { i93, i1 } @llvm.sadd.with.overflow.i93(i93 %[[LOAD1]], i93 %[[LOAD2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i93, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i93, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallestE - SmallestE;
  // CHECK: %[[LOAD1:.+]] = load i4, i4*
  // CHECK: %[[LOAD2:.+]] = load i4, i4*
  // CHECK: %[[OFCALL:.+]] = call { i4, i1 } @llvm.ssub.with.overflow.i4(i4 %[[LOAD1]], i4 %[[LOAD2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i4, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i4, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_sub_overflow_abort

  JustRightE * JustRightE;
  // CHECK: %[[LOAD1:.+]] = load i31, i31*
  // CHECK: %[[LOAD2:.+]] = load i31, i31*
  // CHECK: %[[OFCALL:.+]] = call { i31, i1 } @llvm.smul.with.overflow.i31(i31 %[[LOAD1]], i31 %[[LOAD2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i31, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i31, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_mul_overflow_abort
}

// CHECK: define void @_Z23UnsignedIntegerOverflowjU7_ExtIntILi23EEjU7_ExtIntILi35EEj
void UnsignedIntegerOverflow(unsigned u,
                             unsigned _ExtInt(23) SmallE,
                             unsigned _ExtInt(35) BigE) {
  u = SmallE + SmallE;
  // CHECK: %[[BIGGESTEADDR:.+]] = alloca i23
  // CHECK: %[[LOADE1:.+]] = load i23, i23* %[[BIGGESTEADDR]]
  // CHECK: %[[LOADE2:.+]] = load i23, i23* %[[BIGGESTEADDR]]
  // CHECK: %[[OFCALL:.+]] = call { i23, i1 } @llvm.uadd.with.overflow.i23(i23 %[[LOADE1]], i23 %[[LOADE2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallE = u + u;
  // CHECK: %[[LOADU1:.+]] = load i32, i32*
  // CHECK: %[[LOADU2:.+]] = load i32, i32*
  // CHECK: %[[OFCALL:.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %[[LOADU1]], i32 %[[LOADU2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i32, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i32, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallE = SmallE + SmallE;
  // CHECK: %[[LOADE1:.+]] = load i23, i23*
  // CHECK: %[[LOADE2:.+]] = load i23, i23*
  // CHECK: %[[OFCALL:.+]] = call { i23, i1 } @llvm.uadd.with.overflow.i23(i23 %[[LOADE1]], i23 %[[LOADE2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallE = BigE + BigE;
  // CHECK: %[[LOADE1:.+]] = load i35, i35*
  // CHECK: %[[LOADE2:.+]] = load i35, i35*
  // CHECK: %[[OFCALL:.+]] = call { i35, i1 } @llvm.uadd.with.overflow.i35(i35 %[[LOADE1]], i35 %[[LOADE2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  BigE = BigE + BigE;
  // CHECK: %[[LOADE1:.+]] = load i35, i35*
  // CHECK: %[[LOADE2:.+]] = load i35, i35*
  // CHECK: %[[OFCALL:.+]] = call { i35, i1 } @llvm.uadd.with.overflow.i35(i35 %[[LOADE1]], i35 %[[LOADE2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort
}
