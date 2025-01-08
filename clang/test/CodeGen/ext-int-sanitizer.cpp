// RUN: %clang_cc1 -triple x86_64-gnu-linux -fsanitize=array-bounds,enum,float-cast-overflow,integer-divide-by-zero,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,unsigned-integer-overflow,signed-integer-overflow,shift-base,shift-exponent -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s


// CHECK: define{{.*}} void @_Z6BoundsRA10_KiDB15_
void Bounds(const int (&Array)[10], _BitInt(15) Index) {
  int I1 = Array[Index];
  // CHECK: %[[SEXT:.+]] = sext i15 %{{.+}} to i64
  // CHECK: %[[CMP:.+]] = icmp ult i64 %[[SEXT]], 10
  // CHECK: br i1 %[[CMP]]
  // CHECK: call void @__ubsan_handle_out_of_bounds
}

// CHECK: define{{.*}} void @_Z4Enumv
void Enum() {
  enum E1 { e1a = 0, e1b = 127 }
  e1;
  enum E2 { e2a = -1, e2b = 64 }
  e2;
  enum E3 { e3a = (1u << 31) - 1 }
  e3;

  _BitInt(34) a = e1;
  // CHECK: %[[E1:.+]] = icmp ule i32 %{{.*}}, 127
  // CHECK: br i1 %[[E1]]
  // CHECK: call void @__ubsan_handle_load_invalid_value_abort
  _BitInt(34) b = e2;
  // CHECK: %[[E2HI:.*]] = icmp sle i32 {{.*}}, 127
  // CHECK: %[[E2LO:.*]] = icmp sge i32 {{.*}}, -128
  // CHECK: %[[E2:.*]] = and i1 %[[E2HI]], %[[E2LO]]
  // CHECK: br i1 %[[E2]]
  // CHECK: call void @__ubsan_handle_load_invalid_value_abort
  _BitInt(34) c = e3;
  // CHECK: %[[E3:.*]] = icmp ule i32 {{.*}}, 2147483647
  // CHECK: br i1 %[[E3]]
  // CHECK: call void @__ubsan_handle_load_invalid_value_abort
}

// CHECK: define{{.*}} void @_Z13FloatOverflowfd
void FloatOverflow(float f, double d) {
  _BitInt(10) E = f;
  // CHECK: fcmp ogt float %{{.+}}, -5.130000e+02
  // CHECK: fcmp olt float %{{.+}}, 5.120000e+02
  _BitInt(10) E2 = d;
  // CHECK: fcmp ogt double %{{.+}}, -5.130000e+02
  // CHECK: fcmp olt double %{{.+}}, 5.120000e+02
  _BitInt(7) E3 = f;
  // CHECK: fcmp ogt float %{{.+}}, -6.500000e+01
  // CHECK: fcmp olt float %{{.+}}, 6.400000e+01
  _BitInt(7) E4 = d;
  // CHECK: fcmp ogt double %{{.+}}, -6.500000e+01
  // CHECK: fcmp olt double %{{.+}}, 6.400000e+01
}

// CHECK: define{{.*}} void @_Z14UIntTruncationDU35_jy
void UIntTruncation(unsigned _BitInt(35) E, unsigned int i, unsigned long long ll) {

  i = E;
  // CHECK: %[[LOADE:.+]] = load i64
  // CHECK: %[[E1:.+]] = trunc i64 %[[LOADE]] to i35
  // CHECK: %[[STOREDV:.+]] = zext i35 %[[E1]] to i64
  // CHECK: store i64 %[[STOREDV]], ptr %[[EADDR:.+]]
  // CHECK: %[[LOADE2:.+]] = load i64, ptr %[[EADDR]]
  // CHECK: %[[LOADEDV:.+]] = trunc i64 %[[LOADE2]] to i35
  // CHECK: %[[CONV:.+]] = trunc i35 %[[LOADEDV]] to i32
  // CHECK: %[[EXT:.+]] = zext i32 %[[CONV]] to i35
  // CHECK: %[[CHECK:.+]] = icmp eq i35 %[[EXT]], %[[LOADEDV]]
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

// CHECK: define{{.*}} void @_Z13IntTruncationDB35_DU42_ij
void IntTruncation(_BitInt(35) E, unsigned _BitInt(42) UE, int i, unsigned j) {

  j = E;
  // CHECK: %[[LOADE:.+]] = load i64
  // CHECK: %[[E1:.+]] = trunc i64 %[[LOADE]] to i35
  // CHECK: %[[STOREDV:.+]] = sext i35 %[[E1]] to i64
  // CHECK: store i64 %[[STOREDV]], ptr %[[EADDR:.+]]
  // CHECK: %[[LOADE2:.+]] = load i64, ptr %[[EADDR]]
  // CHECK: %[[LOADEDV:.+]] = trunc i64 %[[LOADE2]] to i35
  // CHECK: %[[CONV:.+]] = trunc i35 %[[LOADEDV]] to i32
  // CHECK: %[[EXT:.+]] = zext i32 %[[CONV]] to i35
  // CHECK: %[[CHECK:.+]] = icmp eq i35 %[[EXT]], %[[LOADEDV]]
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  j = UE;
  // CHECK: %[[LOADUE:.+]] = load i64
  // CHECK: %[[LOADEDV:.+]] = trunc i64 %[[LOADUE]] to i42
  // CHECK: %[[CONV:.+]] = trunc i42 %[[LOADEDV]] to i32
  // CHECK: %[[EXT:.+]] = zext i32 %[[CONV]] to i42
  // CHECK: %[[CHECK:.+]] = icmp eq i42 %[[EXT]], %[[LOADEDV]]
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  // Note: also triggers sign change check.
  i = UE;
  // CHECK: %[[LOADUE:.+]] = load i64
  // CHECK: %[[LOADEDV:.+]] = trunc i64 %[[LOADUE]] to i42
  // CHECK: %[[CONV:.+]] = trunc i42 %[[LOADEDV]] to i32
  // CHECK: %[[NEG:.+]] = icmp slt i32 %[[CONV]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 false, %[[NEG]]
  // CHECK: %[[EXT:.+]] = sext i32 %[[CONV]] to i42
  // CHECK: %[[CHECK:.+]] = icmp eq i42 %[[EXT]], %[[LOADEDV]]
  // CHECK: %[[CHECKBOTH:.+]] = and i1 %[[SIGNCHECK]], %[[CHECK]]
  // CHECK: br i1 %[[CHECKBOTH]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  // Note: also triggers sign change check.
  E = UE;
  // CHECK: %[[LOADUE:.+]] = load i64
  // CHECK: %[[LOADEDV:.+]] = trunc i64 %[[LOADUE]] to i42
  // CHECK: %[[CONV:.+]] = trunc i42 %[[LOADEDV]] to i35
  // CHECK: %[[NEG:.+]] = icmp slt i35 %[[CONV]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 false, %[[NEG]]
  // CHECK: %[[EXT:.+]] = sext i35 %[[CONV]] to i42
  // CHECK: %[[CHECK:.+]] = icmp eq i42 %[[EXT]], %[[LOADEDV]]
  // CHECK: %[[CHECKBOTH:.+]] = and i1 %[[SIGNCHECK]], %[[CHECK]]
  // CHECK: br i1 %[[CHECKBOTH]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort
}

// CHECK: define{{.*}} void @_Z15SignChangeCheckDU39_DB39_
void SignChangeCheck(unsigned _BitInt(39) UE, _BitInt(39) E) {
  UE = E;
  // CHECK: %[[LOADEU:.+]] = load i64
  // CHECK: %[[LOADE:.+]] = load i64
  // CHECK: %[[LOADEDV:.+]] = trunc i64 %[[LOADE]] to i39
  // CHECK: %[[STOREDV:.+]] = sext i39 %[[LOADEDV]] to i64
  // CHECK: store i64 %[[STOREDV]], ptr %[[EADDR:.+]]
  // CHECK: %[[LOADE2:.+]] = load i64, ptr %[[EADDR]]
  // CHECK: %[[LOADEDV2:.+]] = trunc i64 %[[LOADE2]] to i39
  // CHECK: %[[NEG:.+]] = icmp slt i39 %[[LOADEDV2]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 %[[NEG]], false
  // CHECK: br i1 %[[SIGNCHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort

  E = UE;
  // CHECK: %[[STOREDV2:.+]] = zext i39 %[[LOADEDV2]] to i64
  // CHECK: store i64 %[[STOREDV2]], ptr %[[UEADDR:.+]]
  // CHECK: %[[LOADUE2:.+]] = load i64, ptr %[[UEADDR]]
  // CHECK: %[[LOADEDV3:.+]] = trunc i64 %[[LOADUE2]] to i39
  // CHECK: %[[NEG:.+]] = icmp slt i39 %[[LOADEDV3]], 0
  // CHECK: %[[SIGNCHECK:.+]] = icmp eq i1 false, %[[NEG]]
  // CHECK: br i1 %[[SIGNCHECK]]
  // CHECK: call void @__ubsan_handle_implicit_conversion_abort
}

// CHECK: define{{.*}} void @_Z9DivByZeroDB11_i
void DivByZero(_BitInt(11) E, int i) {

  // Also triggers signed integer overflow.
  E / E;
  // CHECK: %[[EADDR:.+]] = alloca i16
  // CHECK: %[[E:.+]] = load i16, ptr %[[EADDR]]
  // CHECK: %[[LOADEDE:.+]] = trunc i16 %[[E]] to i11
  // CHECK: %[[E2:.+]] = load i16, ptr %[[EADDR]]
  // CHECK: %[[LOADEDE2:.+]] = trunc i16 %[[E2]] to i11
  // CHECK: %[[NEZERO:.+]] = icmp ne i11 %[[LOADEDE2]], 0
  // CHECK: %[[NEMIN:.+]] = icmp ne i11 %[[LOADEDE]], -1024
  // CHECK: %[[NENEG1:.+]] = icmp ne i11 %[[LOADEDE2]], -1
  // CHECK: %[[OR:.+]] = or i1 %[[NEMIN]], %[[NENEG1]]
  // CHECK: %[[AND:.+]] = and i1 %[[NEZERO]], %[[OR]]
  // CHECK: br i1 %[[AND]]
  // CHECK: call void @__ubsan_handle_divrem_overflow_abort
}

// TODO:
//-fsanitize=shift: (shift-base, shift-exponent) Shift operators where the amount shifted is greater or equal to the promoted bit-width of the left hand side or less than zero, or where the left hand side is negative. For a signed left shift, also checks for signed overflow in C, and for unsigned overflow in C++. You can use -fsanitize=shift-base or -fsanitize=shift-exponent to check only left-hand side or right-hand side of shift operation, respectively.
// CHECK: define{{.*}} void @_Z6ShiftsDB9_
void Shifts(_BitInt(9) E) {
  E >> E;
  // CHECK: %[[EADDR:.+]] = alloca i16
  // CHECK: %[[LHSE:.+]] = load i16, ptr %[[EADDR]]
  // CHECK: %[[RHSE:.+]] = load i16, ptr %[[EADDR]]
  // CHECK: %[[LOADED:.+]] = trunc i16 %[[RHSE]] to i9
  // CHECK: %[[CMP:.+]] = icmp ule i9 %[[LOADED]], 8
  // CHECK: br i1 %[[CMP]]
  // CHECK: call void @__ubsan_handle_shift_out_of_bounds_abort

  E << E;
  // CHECK: %[[LHSE:.+]] = load i16, ptr
  // CHECK: %[[LOADEDL:.+]] = trunc i16 %[[LHSE]] to i9
  // CHECK: %[[RHSE:.+]] = load i16, ptr
  // CHECK: %[[LOADED:.+]] = trunc i16 %[[RHSE]] to i9
  // CHECK: %[[CMP:.+]] = icmp ule i9 %[[LOADED]], 8
  // CHECK: br i1 %[[CMP]]
  // CHECK: %[[ZEROS:.+]] = sub nuw nsw i9 8, %[[LOADED]]
  // CHECK: %[[CHECK:.+]] = lshr i9 %[[LOADEDL]], %[[ZEROS]]
  // CHECK: %[[SKIPSIGN:.+]] = lshr i9 %[[CHECK]], 1
  // CHECK: %[[CHECK:.+]] = icmp eq i9 %[[SKIPSIGN]]
  // CHECK: %[[PHI:.+]] = phi i1 [ true, %{{.+}} ], [ %[[CHECK]], %{{.+}} ]
  // CHECK: and i1 %[[CMP]], %[[PHI]]
  // CHECK: call void @__ubsan_handle_shift_out_of_bounds_abort
}

// CHECK: define{{.*}} void @_Z21SignedIntegerOverflowDB93_DB4_DB31_
void SignedIntegerOverflow(_BitInt(93) BiggestE,
                           _BitInt(4) SmallestE,
                           _BitInt(31) JustRightE) {
  BiggestE + BiggestE;
  // CHECK: %[[LOADBIGGESTE2:.+]] = load i128
  // CHECK: %[[LOADEDV:.+]] = trunc i128 %[[LOADBIGGESTE2]] to i93
  // CHECK: %[[STOREDV:.+]] = sext i93 %[[LOADEDV]] to i128
  // CHECK: store i128 %[[STOREDV]], ptr %[[BIGGESTEADDR:.+]]
  // CHECK: %[[LOAD1:.+]] = load i128, ptr %[[BIGGESTEADDR]]
  // CHECK: %[[LOADEDV1:.+]] = trunc i128 %[[LOAD1]] to i93
  // CHECK: %[[LOAD2:.+]] = load i128, ptr %[[BIGGESTEADDR]]
  // CHECK: %[[LOADEDV2:.+]] = trunc i128 %[[LOAD2]] to i93
  // CHECK: %[[OFCALL:.+]] = call { i93, i1 } @llvm.sadd.with.overflow.i93(i93 %[[LOADEDV1]], i93 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i93, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i93, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallestE - SmallestE;
  // CHECK: %[[LOAD1:.+]] = load i8, ptr
  // CHECK: %[[LOADEDV1:.+]] = trunc i8 %[[LOAD1]] to i4
  // CHECK: %[[LOAD2:.+]] = load i8, ptr
  // CHECK: %[[LOADEDV2:.+]] = trunc i8 %[[LOAD2]] to i4
  // CHECK: %[[OFCALL:.+]] = call { i4, i1 } @llvm.ssub.with.overflow.i4(i4 %[[LOADEDV1]], i4 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i4, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i4, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_sub_overflow_abort

  JustRightE * JustRightE;
  // CHECK: %[[LOAD1:.+]] = load i32, ptr
  // CHECK: %[[LOADEDV1:.+]] = trunc i32 %[[LOAD1]] to i31
  // CHECK: %[[LOAD2:.+]] = load i32, ptr
  // CHECK: %[[LOADEDV2:.+]] = trunc i32 %[[LOAD2]] to i31
  // CHECK: %[[OFCALL:.+]] = call { i31, i1 } @llvm.smul.with.overflow.i31(i31 %[[LOADEDV1]], i31 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i31, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i31, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_mul_overflow_abort
}

// CHECK: define{{.*}} void @_Z23UnsignedIntegerOverflowjDU23_DU35_
void UnsignedIntegerOverflow(unsigned u,
                             unsigned _BitInt(23) SmallE,
                             unsigned _BitInt(35) BigE) {
  u = SmallE + SmallE;
  // CHECK: %[[LOADE1:.+]] = load i32, ptr
  // CHECK-NEXT: %[[LOADEDV1:.+]] = trunc i32 %[[LOADE1]] to i23
  // CHECK: %[[LOADE2:.+]] = load i32, ptr
  // CHECK-NEXT: %[[LOADEDV2:.+]] = trunc i32 %[[LOADE2]] to i23
  // CHECK: %[[OFCALL:.+]] = call { i23, i1 } @llvm.uadd.with.overflow.i23(i23 %[[LOADEDV1]], i23 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallE = u + u;
  // CHECK: %[[LOADU1:.+]] = load i32, ptr
  // CHECK: %[[LOADU2:.+]] = load i32, ptr
  // CHECK: %[[OFCALL:.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %[[LOADU1]], i32 %[[LOADU2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i32, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i32, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallE = SmallE + SmallE;
  // CHECK: %[[LOADE1:.+]] = load i32, ptr
  // CHECK-NEXT: %[[LOADEDV1:.+]] = trunc i32 %[[LOADE1]] to i23
  // CHECK: %[[LOADE2:.+]] = load i32, ptr
  // CHECK-NEXT: %[[LOADEDV2:.+]] = trunc i32 %[[LOADE2]] to i23
  // CHECK: %[[OFCALL:.+]] = call { i23, i1 } @llvm.uadd.with.overflow.i23(i23 %[[LOADEDV1]], i23 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i23, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  SmallE = BigE + BigE;
  // CHECK: %[[LOADE1:.+]] = load i64, ptr
  // CHECK-NEXT: %[[LOADEDV1:.+]] = trunc i64 %[[LOADE1]] to i35
  // CHECK: %[[LOADE2:.+]] = load i64, ptr
  // CHECK-NEXT: %[[LOADEDV2:.+]] = trunc i64 %[[LOADE2]] to i35
  // CHECK: %[[OFCALL:.+]] = call { i35, i1 } @llvm.uadd.with.overflow.i35(i35 %[[LOADEDV1]], i35 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort

  BigE = BigE + BigE;
  // CHECK: %[[LOADE1:.+]] = load i64, ptr
  // CHECK-NEXT: %[[LOADEDV1:.+]] = trunc i64 %[[LOADE1]] to i35
  // CHECK: %[[LOADE2:.+]] = load i64, ptr
  // CHECK-NEXT: %[[LOADEDV2:.+]] = trunc i64 %[[LOADE2]] to i35
  // CHECK: %[[OFCALL:.+]] = call { i35, i1 } @llvm.uadd.with.overflow.i35(i35 %[[LOADEDV1]], i35 %[[LOADEDV2]])
  // CHECK: %[[EXRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 0
  // CHECK: %[[OFRESULT:.+]] = extractvalue { i35, i1 } %[[OFCALL]], 1
  // CHECK: %[[CHECK:.+]] = xor i1 %[[OFRESULT]], true
  // CHECK: br i1 %[[CHECK]]
  // CHECK: call void @__ubsan_handle_add_overflow_abort
}
