#include "C2000.h"
#include "Targets.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::targets;

const char *const C2000TargetInfo::GCCRegNames[] = {
    "ACC", "XAR0", "XAR1", "XAR2", "XAR3", "XAR4", "XAR5", "XAR6", "XAR7"};

ArrayRef<const char *> C2000TargetInfo::getGCCRegNames() const {
  return llvm::ArrayRef(GCCRegNames);
}

bool C2000TargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                           DiagnosticsEngine &Diags) {

  for (const auto &Feature : Features) {
    if (Feature == "+eabi") {
      eabi = true;
      continue;
    }
    if (Feature == "+strict_ansi") {
      strict = true;
      continue;
    }
    if (Feature == "+cla_support") {
      cla_support = true;
    }
    if (Feature == "+cla0") {
      cla0 = true;
      continue;
    }
    if (Feature == "+cla1") {
      cla1 = true;
      continue;
    }
    if (Feature == "+cla2") {
      cla2 = true;
      continue;
    }
    if (Feature == "+relaxed") {
      relaxed = true;
      continue;
    }
    if (Feature == "+fpu64") {
      fpu64 = true;
      continue;
    }
    if (Feature == "+fpu32") {
      fpu32 = true;
      continue;
    }
    if (Feature == "+tmu_support") {
      tmu_support = true;
    }
    if (Feature == "+tmu1") {
      tmu1 = true;
      continue;
    }
    if (Feature == "+idiv0") {
      idiv0 = true;
      continue;
    }
    if (Feature == "+vcu_support") {
      vcu_support = true;
    }
    if (Feature == "+vcu2") {
      vcu2 = true;
      continue;
    }
    if (Feature == "+vcrc") {
      vcrc = true;
      continue;
    }
    if (Feature == "+opt_level") {
      opt = true;
      continue;
    }
  }
  return true;
}

bool C2000TargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("eabi", eabi)
      .Case("strict_ansi", strict)
      .Case("cla-support", cla_support)
      .Case("cla0", cla0)
      .Case("cla1", cla1)
      .Case("cla2", cla2)
      .Case("relaxed", relaxed)
      .Case("fpu64", fpu64)
      .Case("fpu32", fpu32)
      .Case("tmu-support", tmu_support)
      .Case("tmu1", tmu1)
      .Case("vcu-support", vcu_support)
      .Case("vcu2", vcu2)
      .Case("vcrc", vcrc)
      .Case("opt-level", opt)
      .Default(false);
}

void C2000TargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.undefineMacro("__CHAR_BIT__"); // FIXME: Implement 16-bit char
  Builder.defineMacro("__CHAR_BIT__", "16");
  Builder.defineMacro("__TMS320C2000__");
  Builder.defineMacro("_TMS320C2000");
  Builder.defineMacro("__TMS320C28XX__");
  Builder.defineMacro("_TMS320C28XX");
  Builder.defineMacro("__TMS320C28X__");
  Builder.defineMacro("_TMS320C28X");
  Builder.defineMacro("__TI_STRICT_FP_MODE__");
  Builder.defineMacro("__COMPILER_VERSION__");
  Builder.defineMacro("__TI_COMPILER_VERSION__");
  Builder.defineMacro("__TI_COMPILER_VERSION__QUAL_ID");
  Builder.defineMacro("__TI_COMPILER_VERSION__QUAL__", "QUAL_LETTER");
  Builder.defineMacro("__little_endian__");
  Builder.defineMacro("__PTRDIFF_T_TYPE__", "signed long");
  Builder.defineMacro("__SIZE_T_TYPE__", "unsigned long");
  Builder.defineMacro("__WCHAR_T_TYPE__", "long unsigned");
  Builder.defineMacro("__TI_WCHAR_T_BITS", "16");
  Builder.defineMacro("__TI_C99_COMPLEX_ENABLED");
  Builder.defineMacro("__TI_GNU_ATTRIBUTE_SUPPORT__");
  Builder.defineMacro("__LARGE_MODEL__");
  Builder.defineMacro("__signed_chars__");
  Builder.defineMacro("__OPTIMIZE_FOR_SPACE");

  if (hasFeature("eabi"))
    Builder.defineMacro("__TI_EABI__");
  if (hasFeature("strict_ansi"))
    Builder.defineMacro("__TI_STRICT_ANSI_MODE__");
  if (hasFeature("cla-support"))
    Builder.defineMacro("__TMS320C28XX_CLA__");

  if (hasFeature("cla0"))
    Builder.defineMacro("__TMS320C28XX_CLA0__");
  else if (hasFeature("cla1"))
    Builder.defineMacro("__TMS320C28XX_CLA1__");
  else if (hasFeature("cla2"))
    Builder.defineMacro("__TMS320C28XX_CLA2__");

  if (hasFeature("fpu64")) {
    Builder.defineMacro("__TMS320C28XX_FPU64__");
    Builder.defineMacro("__TMS320C28XX_FPU32__");
  } else if (hasFeature("fpu32"))
    Builder.defineMacro("__TMS320C28XX_FPU32__");
  if (hasFeature("idiv0"))
    Builder.defineMacro("__TMS320C28XX_IDIV__");
  if (hasFeature("tmu1"))
    Builder.defineMacro("__TMS320C28XX_TMU1__");
  if (hasFeature("tmu-support")) {
    Builder.defineMacro("__TMS320C28XX_TMU0__");
    Builder.defineMacro("__TMS320C28XX_TMU__");
  }
  if (hasFeature("vcu-support"))
    Builder.defineMacro("__TMS320C28XX_VCU0__");
  if (hasFeature("vcu2"))
    Builder.defineMacro("__TMS320C28XX_VCU2__");
  else if (hasFeature("vcrc"))
    Builder.defineMacro("__TMS320C28XX_VCRC__");
  if (hasFeature("opt-level"))
    Builder.defineMacro("_INLINE");
  if (hasFeature("relaxed"))
    Builder.undefineMacro("__TI_STRICT_FP_MODE__");

  Builder.defineMacro("__cregister", "");
  Builder.defineMacro("interrupt", "");
  Builder.defineMacro("__interrupt", "");

  // Assembly Instrinsics

  Builder.append("int __abs16_sat( int src );");
  Builder.append("void __add( int *m, int b );");
  Builder.append("long __addcu( long src1, unsigned int src2 );");
  Builder.append("void __addl( long *m, long b );");
  Builder.append("void __and(int *m, int b);");
  Builder.append("int *__byte_func( int *array, unsigned int byte_index );");
  Builder.defineMacro("__byte(array, byte_index)",
                      "*__byte_func(array, byte_index)");
  Builder.append("unsigned long *__byte_peripheral_32_func(unsigned long *x);");
  Builder.defineMacro("__byte_peripheral_32(x)",
                      "*__byte_peripheral_32_func(x)");
  Builder.append("void __dec( int *m );");

  // dmac needs macro magic
  Builder.append("unsigned int __disable_interrupts( );");
  Builder.append("void __eallow( void );");
  Builder.append("void __edis( void );");
  Builder.append("unsigned int __enable_interrupts( );");
  Builder.append("int __flip16(int src);");
  Builder.append("long __flip32(long src);");
  Builder.append("long long __flip64(long long src);");
  Builder.append("void __inc( int *m );");
  Builder.append("long __IQ( long double A , int N );");
  Builder.append("long __IQmpy( long A, long B , int N );");
  Builder.append("long __IQsat( long A, long max, long min );");
  Builder.append("long __IQxmpy(long A , long B, int N);");
  Builder.append("long long __llmax(long long dst, long long src);");
  Builder.append("long long __llmin(long long dst, long long src);");
  Builder.append("long __lmax(long dst, long src);");
  Builder.append("long __lmin(long dst, long src);");
  Builder.append("int __max(int dst, int src);");
  Builder.append("int __min(int dst, int src);");
  Builder.append("int __mov_byte( int *src, unsigned int n );");
  Builder.append("long __mpy( int src1, int src2 );");
  Builder.append("long __mpyb( int src1, unsigned int src2 );");
  Builder.append("long __mpy_mov_t( int src1, int src2, int * dst2 );");
  Builder.append("unsigned long __mpyu(unsigned int src2, unsigned int srt2);");
  Builder.append("long __mpyxu( int src1, unsigned int src2 );");
  Builder.append("long __norm32(long src, int * shift );");
  Builder.append("long long __norm64(long long src, int * shift );");
  Builder.append("void __or(int *m, int b);");
  Builder.append("long __qmpy32( long src32a, long src32b, int q );");
  Builder.append("long __qmpy32by16(long src32, int src16, int q);");
  Builder.append("void __restore_interrupts(unsigned int val );");
  Builder.append("long __rol( long src );");
  Builder.append("long __ror( long src );");
  Builder.append("void * __rpt_mov_imm(void * dst , int src ,int count );");
  Builder.append("int __rpt_norm_inc( long src, int dst, int count );");
  Builder.append("int __rpt_norm_dec(long src, int dst, int count);");
  Builder.append("long __rpt_rol(long src, int count);");
  Builder.append("long __rpt_ror(long src, int count);");
  Builder.append("long __rpt_subcu(long dst, int src, int count);");
  Builder.append("unsigned long __rpt_subcul(unsigned long num, unsigned long "
                 "den, unsigned long remainder, int count);");
  Builder.append("long __sat( long src );");
  Builder.append("long __sat32( long src, long limit );");
  Builder.append("long __sathigh16(long src, int limit);");
  Builder.append("long __satlow16( long src );");
  Builder.append("long __sbbu( long src1 , unsigned int src2 );");
  Builder.append("void __sub( int * m, int b );");
  Builder.append("long __subcu( long src1, int src2 );");
  Builder.append("unsigned long __subcul(unsigned long num, unsigned long den, "
                 "unsigned long remainder);");
  Builder.append("void __subl( long * m, long b );");
  Builder.append("void __subr( int * m , int b );");
  Builder.append("void __subrl( long * m , long b );");
  Builder.append("int __tbit( int src , int bit );");
  Builder.append("void __xor( int * m, int b );");

  // FPU Intrinsics
  if (hasFeature("fpu64")) {
    Builder.append("double __einvf64( double x );");
    Builder.append("double __eisqrtf64( double x );");
    Builder.append("void __f64_max_idx( double dst, double src, double "
                   "idx_dst, double idx_src );");
    Builder.append("void __f64_min_idx( double dst, double src, double "
                   "idx_dst, double idx_src );");
    Builder.append("double __fmax64( double x, double y );");
    Builder.append("double __fmin64( double x, double y );");
    Builder.append("double __fracf64(double src );");
    Builder.append("void __swapff( double &a, double &b );");
  } else {
    Builder.append("float __eisqrtf32( float x );");
    Builder.append("void __f32_max_idx( float dst, float src, float idx_dst, "
                   "float idx_src );");
    Builder.append("void __f32_min_idx( float dst, float src, float idx_dst, "
                   "float idx_src );");
    Builder.append("int __f32toi16r(float src );");
    Builder.append("unsigned int __f32toui16r(float src );");
    Builder.append("float __fmax( float x, float y );");
    Builder.append("float __fmin( float x, float y );");
    Builder.append("float __fracf32(float src );");
    Builder.append("float __fsat(float val, float max, float min );");
    Builder.append("void __swapf( float &a, float &b );");
    Builder.append("void __swapff( float &a, float &b );");
  }

  // Trigonometric Math Unit (TMU) Intrinsics
  if (hasFeature("tmu-support")) {
    Builder.append("float __atan( float src );");
    Builder.append("float __atan2( float y , float x );");
    Builder.append("float __atanpuf32( float src );");
    Builder.append("float __atan2puf32( float x, float y );");
    Builder.append("float __cos( float src );");
    Builder.append("float __cospuf32( float src );");
    Builder.append("float __divf32( float num , float denom );");
    Builder.append("float __div2pif32( float src );");
    Builder.append("float __mpy2pif32( float src );");
    Builder.append("float __quadf32( float ratio, float y, float x );");
    Builder.append("float __sin( float src );");
    Builder.append("float __sinpuf32( float src );");
    Builder.append("float __sqrt( float src );");
  }

  // Fast Integer Division Intrinsics
  if (hasFeature("idiv-support")) {
    Builder.append(
        "ldiv_t __traditional_div_i16byi16( int dividend, int divisor );");
    Builder.append(
        "ldiv_t __euclidean_div_i16byi16( int dividend, int divisor );");
    Builder.append(
        "ldiv_t __modulo_div_i16byi16( int dividend, int divisor );");
    Builder.append("_ulldiv_t __traditional_div_u16byu16( unsigned int "
                   "dividend, unsigned int divisor );");
    Builder.append(
        "ldiv_t __traditional_div_i32byi32( long dividend, long divisor );");
    Builder.append(
        "ldiv_t __euclidean_div_i32byi32( long dividend, long divisor );");
    Builder.append(
        "ldiv_t __modulo_div_i32byi32( long dividend, long divisor );");
    Builder.append("ldiv_t __traditional_div_i32byu32( long dividend, unsigned "
                   "long divisor );");
    Builder.append("ldiv_t __euclidean_div_i32byu32( long dividend, unsigned "
                   "long divisor );");
    Builder.append("ldiv_t __modulo_div_i32byu32( long dividend, unsigned long "
                   "divisor );");
    Builder.append("_ulldiv_t __traditional_div_u32byu32( unsigned long "
                   "dividend, unsigned long divisor );");
    Builder.append(
        "ldiv_t __traditional_div_i32byi16( long dividend, int divisor );");
    Builder.append(
        "ldiv_t __euclidean_div_i32byi16( long dividend, int divisor );");
    Builder.append(
        "ldiv_t __modulo_div_i32byi16( long dividend, int divisor );");
    Builder.append("lldiv_t __traditional_div_i64byi64( long long dividend, "
                   "long long divisor );");
    Builder.append("lldiv_t __euclidean_div_i64byi64( long long dividend, long "
                   "long divisor );");
    Builder.append("lldiv_t __modulo_div_i64byi64( long long dividend, long "
                   "long divisor );");
    Builder.append("lldiv_t __traditional_div_i64byu64( long long dividend, "
                   "unsigned long long divisor );");
    Builder.append("lldiv_t __euclidean_div_i64byu64( long long dividend, "
                   "unsigned long long divisor );");
    Builder.append("lldiv_t __modulo_div_i64byu64( long long dividend, "
                   "unsigned long long divisor );");
    Builder.append("__ulldiv_t __traditional_div_u64byu64( unsigned long long "
                   "dividend, unsigned long long divisor );");
    Builder.append("lldiv_t __traditional_div_i64byi32( signed long long "
                   "dividend, long divisor);");
    Builder.append("lldiv_t __euclidean_div_i64byi32( signed long long "
                   "dividend, long divisor );");
    Builder.append("lldiv_t __modulo_div_i64byi32( signed long long dividend, "
                   "long divisor );");
    Builder.append("lldiv_t __traditional_div_i64byu32( signed long long "
                   "dividend, unsigned long divisor );");
    Builder.append("lldiv_t __euclidean_div_i64byu32( signed long long "
                   "dividend, unsigned long divisor );");
    Builder.append("lldiv_t __modulo_div_i64byu32( unsigned long long "
                   "dividend, unsigned long divisor );");
    Builder.append("__ulldiv_t __traditional_div_u64byu32( unsigned long long "
                   "dividend, unsigned long divisor );");
  }

  // Non-documented intrinsics
  Builder.append("void *memcpy(void * __restrict s1, const void * __restrict "
                 "s2, unsigned long n);");
  Builder.defineMacro("__intaddr__(x)", "x");
  Builder.defineMacro("asm(x)", "");
}
