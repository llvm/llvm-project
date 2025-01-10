! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program p
  use ieee_arithmetic, only: ieee_value, ieee_negative_inf, ieee_positive_inf
  use ieee_arithmetic, only: ieee_next_after, ieee_next_down, ieee_next_up
  implicit none
  ! CHECK-DAG: %[[V_4:[0-9]+]] = fir.alloca f80 {bindc_name = "r10", uniq_name = "_QFEr10"}
  ! CHECK-DAG: %[[V_5:[0-9]+]] = fir.declare %[[V_4]] {uniq_name = "_QFEr10"} : (!fir.ref<f80>) -> !fir.ref<f80>
  ! CHECK-DAG: %[[V_6:[0-9]+]] = fir.alloca f128 {bindc_name = "r16", uniq_name = "_QFEr16"}
  ! CHECK-DAG: %[[V_7:[0-9]+]] = fir.declare %[[V_6]] {uniq_name = "_QFEr16"} : (!fir.ref<f128>) -> !fir.ref<f128>
  ! CHECK-DAG: %[[V_8:[0-9]+]] = fir.alloca f16 {bindc_name = "r2", uniq_name = "_QFEr2"}
  ! CHECK-DAG: %[[V_9:[0-9]+]] = fir.declare %[[V_8]] {uniq_name = "_QFEr2"} : (!fir.ref<f16>) -> !fir.ref<f16>
  ! CHECK-DAG: %[[V_10:[0-9]+]] = fir.alloca bf16 {bindc_name = "r3", uniq_name = "_QFEr3"}
  ! CHECK-DAG: %[[V_11:[0-9]+]] = fir.declare %[[V_10]] {uniq_name = "_QFEr3"} : (!fir.ref<bf16>) -> !fir.ref<bf16>
  ! CHECK-DAG: %[[V_12:[0-9]+]] = fir.alloca f32 {bindc_name = "r4", uniq_name = "_QFEr4"}
  ! CHECK-DAG: %[[V_13:[0-9]+]] = fir.declare %[[V_12]] {uniq_name = "_QFEr4"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK-DAG: %[[V_14:[0-9]+]] = fir.alloca f64 {bindc_name = "r8", uniq_name = "_QFEr8"}
  ! CHECK-DAG: %[[V_15:[0-9]+]] = fir.declare %[[V_14]] {uniq_name = "_QFEr8"} : (!fir.ref<f64>) -> !fir.ref<f64>
  ! CHECK-DAG: %[[V_16:[0-9]+]] = fir.address_of(@_QFEx10) : !fir.ref<f80>
  ! CHECK-DAG: %[[V_17:[0-9]+]] = fir.declare %[[V_16]] {uniq_name = "_QFEx10"} : (!fir.ref<f80>) -> !fir.ref<f80>
  ! CHECK-DAG: %[[V_18:[0-9]+]] = fir.alloca f128 {bindc_name = "x16", uniq_name = "_QFEx16"}
  ! CHECK-DAG: %[[V_19:[0-9]+]] = fir.declare %[[V_18]] {uniq_name = "_QFEx16"} : (!fir.ref<f128>) -> !fir.ref<f128>
  ! CHECK-DAG: %[[V_20:[0-9]+]] = fir.alloca f16 {bindc_name = "x2", uniq_name = "_QFEx2"}
  ! CHECK-DAG: %[[V_21:[0-9]+]] = fir.declare %[[V_20]] {uniq_name = "_QFEx2"} : (!fir.ref<f16>) -> !fir.ref<f16>
  ! CHECK-DAG: %[[V_22:[0-9]+]] = fir.address_of(@_QFEx3) : !fir.ref<bf16>
  ! CHECK-DAG: %[[V_23:[0-9]+]] = fir.declare %[[V_22]] {uniq_name = "_QFEx3"} : (!fir.ref<bf16>) -> !fir.ref<bf16>
  ! CHECK-DAG: %[[V_24:[0-9]+]] = fir.address_of(@_QFEx4) : !fir.ref<f32>
  ! CHECK-DAG: %[[V_25:[0-9]+]] = fir.declare %[[V_24]] {uniq_name = "_QFEx4"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK-DAG: %[[V_26:[0-9]+]] = fir.address_of(@_QFEx8) : !fir.ref<f64>
  ! CHECK-DAG: %[[V_27:[0-9]+]] = fir.declare %[[V_26]] {uniq_name = "_QFEx8"} : (!fir.ref<f64>) -> !fir.ref<f64>
  real(2)  ::  r2,  x2
  real(3)  ::  r3,  x3 = -huge(x3)
  real(4)  ::  r4,  x4 = -0.
  real(8)  ::  r8,  x8 =  0.
  real(10) :: r10, x10 =  huge(x10)
  real(16) :: r16, x16

  x2  = ieee_value(x2, ieee_negative_inf)
  x16 = ieee_value(x2, ieee_positive_inf)

  ! CHECK:     %[[V_45:[0-9]+]] = fir.load %[[V_21]] : !fir.ref<f16>
  ! CHECK:     %[[V_46:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f80>
  ! CHECK-DAG: %[[V_47:[0-9]+]] = fir.coordinate_of %{{.*}}, %c2{{.*}} : (!fir.ref<!fir.array<12xi16>>, i8) -> !fir.ref<i16>
  ! CHECK-DAG: %[[V_48:[0-9]+]] = fir.load %[[V_47]] : !fir.ref<i16>
  ! CHECK-DAG: %[[V_49:[0-9]+]] = arith.bitcast %[[V_48]] : i16 to f16
  ! CHECK-DAG: %[[V_50:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_46]]) <{bit = 3 : i32}> : (f80) -> i1
  ! CHECK:     %[[V_51:[0-9]+]] = arith.select %[[V_50]], %[[V_49]], %[[V_45]] : f16
  ! CHECK:     %[[V_52:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_51]]) <{bit = 3 : i32}> : (f16) -> i1
  ! CHECK:     %[[V_53:[0-9]+]] = fir.convert %[[V_51]] : (f16) -> f80
  ! CHECK:     %[[V_54:[0-9]+]] = arith.cmpf oeq, %[[V_53]], %[[V_46]] fastmath<contract> : f80
  ! CHECK:     %[[V_55:[0-9]+]] = arith.ori %[[V_52]], %[[V_54]] : i1
  ! CHECK:     %[[V_56:[0-9]+]] = arith.cmpf olt, %[[V_53]], %[[V_46]] fastmath<contract> : f80
  ! CHECK:     %[[V_57:[0-9]+]] = arith.bitcast %[[V_45]] : f16 to i16
  ! CHECK:     %[[V_58:[0-9]+]] = arith.shrui %[[V_57]], %c15{{.*}} : i16
  ! CHECK:     %[[V_59:[0-9]+]] = fir.convert %[[V_58]] : (i16) -> i1
  ! CHECK:     %[[V_60:[0-9]+]] = arith.cmpi ne, %[[V_56]], %[[V_59]] : i1
  ! CHECK:     %[[V_61:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_51]]) <{bit = 516 : i32}> : (f16) -> i1
  ! CHECK:     %[[V_62:[0-9]+]] = arith.andi %[[V_61]], %[[V_60]] : i1
  ! CHECK:     %[[V_63:[0-9]+]] = arith.ori %[[V_55]], %[[V_62]] : i1
  ! CHECK:     %[[V_64:[0-9]+]] = fir.if %[[V_63]] -> (f16) {
  ! CHECK:       fir.result %[[V_51]] : f16
  ! CHECK:     } else {
  ! CHECK:       %[[V_202:[0-9]+]] = arith.cmpf oeq, %[[V_51]], %cst{{[_0-9]*}} fastmath<contract> : f16
  ! CHECK:       %[[V_203:[0-9]+]] = fir.if %[[V_202]] -> (f16) {
  ! CHECK:         %[[V_204:[0-9]+]] = arith.select %[[V_56]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f16
  ! CHECK:         %[[V_205:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                             fir.call @feraiseexcept(%[[V_205]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_204]] : f16
  ! CHECK:       } else {
  ! CHECK:         %[[V_204:[0-9]+]] = arith.bitcast %[[V_51]] : f16 to i16
  ! CHECK-DAG:     %[[V_205:[0-9]+]] = arith.subi %[[V_204]], %c1{{.*}} : i16
  ! CHECK-DAG:     %[[V_206:[0-9]+]] = arith.addi %[[V_204]], %c1{{.*}} : i16
  ! CHECK:         %[[V_207:[0-9]+]] = arith.select %[[V_60]], %[[V_206]], %[[V_205]] : i16
  ! CHECK:         %[[V_208:[0-9]+]] = arith.bitcast %[[V_207]] : i16 to f16
  ! CHECK:         %[[V_209:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_208]]) <{bit = 516 : i32}> : (f16) -> i1
  ! CHECK:         fir.if %[[V_209]] {
  ! CHECK:           %[[V_211:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                               fir.call @feraiseexcept(%[[V_211]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         %[[V_210:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_208]]) <{bit = 144 : i32}> : (f16) -> i1
  ! CHECK:         fir.if %[[V_210]] {
  ! CHECK:           %[[V_211:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                               fir.call @feraiseexcept(%[[V_211]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_208]] : f16
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_203]] : f16
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_64]] to %[[V_9]] : !fir.ref<f16>
  r2 = ieee_next_after(x2, x10)
  print "('after:  ', z4.4, ' -> ', z4.4, ' = ', g0)", x2, r2, r2

  ! CHECK:     %[[V_81:[0-9]+]] = fir.load %[[V_23]] : !fir.ref<bf16>
  ! CHECK:     %[[V_82:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_81]]) <{bit = 3 : i32}> : (bf16) -> i1
  ! CHECK:     %[[V_83:[0-9]+]] = fir.convert %[[V_81]] : (bf16) -> f32
  ! CHECK:     %[[V_84:[0-9]+]] = arith.bitcast %[[V_83]] : f32 to i32
  ! CHECK:     %[[V_85:[0-9]+]] = arith.shrui %[[V_84]], %c31{{.*}} : i32
  ! CHECK:     %[[V_86:[0-9]+]] = fir.convert %[[V_85]] : (i32) -> i1
  ! CHECK:     %[[V_87:[0-9]+]] = arith.cmpi ne, %[[V_86]], %true{{[_0-9]*}} : i1
  ! CHECK:     %[[V_88:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_81]]) <{bit = 516 : i32}> : (bf16) -> i1
  ! CHECK:     %[[V_89:[0-9]+]] = arith.andi %[[V_88]], %[[V_87]] : i1
  ! CHECK:     %[[V_90:[0-9]+]] = arith.ori %[[V_82]], %[[V_89]] : i1
  ! CHECK:     %[[V_91:[0-9]+]] = fir.if %[[V_90]] -> (bf16) {
  ! CHECK:       %[[V_202:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_81]]) <{bit = 1 : i32}> : (bf16) -> i1
  ! CHECK:       fir.if %[[V_202]] {
  ! CHECK:         %[[V_203:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                             fir.call @feraiseexcept(%[[V_203]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_81]] : bf16
  ! CHECK:     } else {
  ! CHECK:       %[[V_202:[0-9]+]] = arith.cmpf oeq, %[[V_81]], %cst{{[_0-9]*}} fastmath<contract> : bf16
  ! CHECK:       %[[V_203:[0-9]+]] = fir.if %[[V_202]] -> (bf16) {
  ! CHECK:         fir.result %cst{{[_0-9]*}} : bf16
  ! CHECK:       } else {
  ! CHECK:         %[[V_204:[0-9]+]] = arith.bitcast %[[V_81]] : bf16 to i16
  ! CHECK-DAG:     %[[V_205:[0-9]+]] = arith.subi %[[V_204]], %c1{{.*}} : i16
  ! CHECK-DAG:     %[[V_206:[0-9]+]] = arith.addi %[[V_204]], %c1{{.*}} : i16
  ! CHECK:         %[[V_207:[0-9]+]] = arith.select %[[V_87]], %[[V_206]], %[[V_205]] : i16
  ! CHECK:         %[[V_208:[0-9]+]] = arith.bitcast %[[V_207]] : i16 to bf16
  ! CHECK:         fir.result %[[V_208]] : bf16
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_203]] : bf16
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_91]] to %[[V_11]] : !fir.ref<bf16>
  r3 = ieee_next_up(x3)
  print "('up:     ', z4.4, ' -> ', z4.4, ' = ', g0)", x3, r3, r3

  ! CHECK:     %[[V_104:[0-9]+]] = fir.load %[[V_25]] : !fir.ref<f32>
  ! CHECK:     %[[V_105:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_104]]) <{bit = 3 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_106:[0-9]+]] = arith.bitcast %[[V_104]] : f32 to i32
  ! CHECK:     %[[V_107:[0-9]+]] = arith.shrui %[[V_106]], %c31{{.*}} : i32
  ! CHECK:     %[[V_108:[0-9]+]] = fir.convert %[[V_107]] : (i32) -> i1
  ! CHECK:     %[[V_109:[0-9]+]] = arith.cmpi ne, %[[V_108]], %false{{[_0-9]*}} : i1
  ! CHECK:     %[[V_110:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_104]]) <{bit = 516 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_111:[0-9]+]] = arith.andi %[[V_110]], %[[V_109]] : i1
  ! CHECK:     %[[V_112:[0-9]+]] = arith.ori %[[V_105]], %[[V_111]] : i1
  ! CHECK:     %[[V_113:[0-9]+]] = fir.if %[[V_112]] -> (f32) {
  ! CHECK:       %[[V_202:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_104]]) <{bit = 1 : i32}> : (f32) -> i1
  ! CHECK:       fir.if %[[V_202]] {
  ! CHECK:         %[[V_203:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                             fir.call @feraiseexcept(%[[V_203]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_104]] : f32
  ! CHECK:     } else {
  ! CHECK:       %[[V_202:[0-9]+]] = arith.cmpf oeq, %[[V_104]], %cst{{[_0-9]*}} fastmath<contract> : f32
  ! CHECK:       %[[V_203:[0-9]+]] = fir.if %[[V_202]] -> (f32) {
  ! CHECK:         fir.result %cst{{[_0-9]*}} : f32
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_204:[0-9]+]] = arith.subi %[[V_106]], %c1{{.*}} : i32
  ! CHECK-DAG:     %[[V_205:[0-9]+]] = arith.addi %[[V_106]], %c1{{.*}} : i32
  ! CHECK:         %[[V_206:[0-9]+]] = arith.select %[[V_109]], %[[V_205]], %[[V_204]] : i32
  ! CHECK:         %[[V_207:[0-9]+]] = arith.bitcast %[[V_206]] : i32 to f32
  ! CHECK:         fir.result %[[V_207]] : f32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_203]] : f32
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_113]] to %[[V_13]] : !fir.ref<f32>
  r4 = ieee_next_down(x4)
  print "('down:   ', z8.8, ' -> ', z8.8, ' = ', g0)", x4, r4, r4

  ! CHECK:     %[[V_125:[0-9]+]] = fir.load %[[V_27]] : !fir.ref<f64>
  ! CHECK:     %[[V_126:[0-9]+]] = fir.load %[[V_21]] : !fir.ref<f16>
  ! CHECK-DAG: %[[V_127:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_8) : !fir.ref<!fir.array<12xi64>>
  ! CHECK-DAG: %[[V_128:[0-9]+]] = fir.coordinate_of %[[V_127]], %c2{{.*}} : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
  ! CHECK-DAG: %[[V_129:[0-9]+]] = fir.load %[[V_128]] : !fir.ref<i64>
  ! CHECK-DAG: %[[V_130:[0-9]+]] = arith.bitcast %[[V_129]] : i64 to f64
  ! CHECK-DAG: %[[V_131:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_126]]) <{bit = 3 : i32}> : (f16) -> i1
  ! CHECK:     %[[V_132:[0-9]+]] = arith.select %[[V_131]], %[[V_130]], %[[V_125]] : f64
  ! CHECK:     %[[V_133:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_132]]) <{bit = 3 : i32}> : (f64) -> i1
  ! CHECK:     %[[V_134:[0-9]+]] = fir.convert %[[V_126]] : (f16) -> f64
  ! CHECK:     %[[V_135:[0-9]+]] = arith.cmpf oeq, %[[V_132]], %[[V_134]] fastmath<contract> : f64
  ! CHECK:     %[[V_136:[0-9]+]] = arith.ori %[[V_133]], %[[V_135]] : i1
  ! CHECK:     %[[V_137:[0-9]+]] = arith.cmpf olt, %[[V_132]], %[[V_134]] fastmath<contract> : f64
  ! CHECK:     %[[V_138:[0-9]+]] = arith.bitcast %[[V_125]] : f64 to i64
  ! CHECK:     %[[V_139:[0-9]+]] = arith.shrui %[[V_138]], %c63{{.*}} : i64
  ! CHECK:     %[[V_140:[0-9]+]] = fir.convert %[[V_139]] : (i64) -> i1
  ! CHECK:     %[[V_141:[0-9]+]] = arith.cmpi ne, %[[V_137]], %[[V_140]] : i1
  ! CHECK:     %[[V_142:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_132]]) <{bit = 516 : i32}> : (f64) -> i1
  ! CHECK:     %[[V_143:[0-9]+]] = arith.andi %[[V_142]], %[[V_141]] : i1
  ! CHECK:     %[[V_144:[0-9]+]] = arith.ori %[[V_136]], %[[V_143]] : i1
  ! CHECK:     %[[V_145:[0-9]+]] = fir.if %[[V_144]] -> (f64) {
  ! CHECK:       fir.result %[[V_132]] : f64
  ! CHECK:     } else {
  ! CHECK:       %[[V_202:[0-9]+]] = arith.cmpf oeq, %[[V_132]], %cst{{[_0-9]*}} fastmath<contract> : f64
  ! CHECK:       %[[V_203:[0-9]+]] = fir.if %[[V_202]] -> (f64) {
  ! CHECK:         %[[V_204:[0-9]+]] = arith.select %[[V_137]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f64
  ! CHECK:         %[[V_205:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                             fir.call @feraiseexcept(%[[V_205]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_204]] : f64
  ! CHECK:       } else {
  ! CHECK:         %[[V_204:[0-9]+]] = arith.bitcast %[[V_132]] : f64 to i64
  ! CHECK-DAG:     %[[V_205:[0-9]+]] = arith.subi %[[V_204]], %c1{{.*}} : i64
  ! CHECK-DAG:     %[[V_206:[0-9]+]] = arith.addi %[[V_204]], %c1{{.*}} : i64
  ! CHECK:         %[[V_207:[0-9]+]] = arith.select %[[V_141]], %[[V_206]], %[[V_205]] : i64
  ! CHECK:         %[[V_208:[0-9]+]] = arith.bitcast %[[V_207]] : i64 to f64
  ! CHECK:         %[[V_209:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_208]]) <{bit = 516 : i32}> : (f64) -> i1
  ! CHECK:         fir.if %[[V_209]] {
  ! CHECK:           %[[V_211:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                               fir.call @feraiseexcept(%[[V_211]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         %[[V_210:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_208]]) <{bit = 144 : i32}> : (f64) -> i1
  ! CHECK:         fir.if %[[V_210]] {
  ! CHECK:           %[[V_211:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                               fir.call @feraiseexcept(%[[V_211]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_208]] : f64
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_203]] : f64
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_145]] to %[[V_15]] : !fir.ref<f64>
  r8 = ieee_next_after(x8, x2)
  print "('after:  ', z16.16, ' -> ', z16.16, ' = ', g0)", x8, r8, r8

  ! CHECK:     %[[V_158:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f80>
  ! CHECK:     %[[V_159:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_158]]) <{bit = 3 : i32}> : (f80) -> i1
  ! CHECK:     %[[V_160:[0-9]+]] = arith.bitcast %[[V_158]] : f80 to i80
  ! CHECK:     %[[V_161:[0-9]+]] = arith.shrui %[[V_160]], %c79{{.*}} : i80
  ! CHECK:     %[[V_162:[0-9]+]] = fir.convert %[[V_161]] : (i80) -> i1
  ! CHECK:     %[[V_163:[0-9]+]] = arith.cmpi ne, %[[V_162]], %true{{[_0-9]*}} : i1
  ! CHECK:     %[[V_164:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_158]]) <{bit = 516 : i32}> : (f80) -> i1
  ! CHECK:     %[[V_165:[0-9]+]] = arith.andi %[[V_164]], %[[V_163]] : i1
  ! CHECK:     %[[V_166:[0-9]+]] = arith.ori %[[V_159]], %[[V_165]] : i1
  ! CHECK:     %[[V_167:[0-9]+]] = fir.if %[[V_166]] -> (f80) {
  ! CHECK:       %[[V_202:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_158]]) <{bit = 1 : i32}> : (f80) -> i1
  ! CHECK:       fir.if %[[V_202]] {
  ! CHECK:         %[[V_203:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                             fir.call @feraiseexcept(%[[V_203]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_158]] : f80
  ! CHECK:     } else {
  ! CHECK:       %[[V_202:[0-9]+]] = arith.cmpf oeq, %[[V_158]], %cst{{[_0-9]*}} fastmath<contract> : f80
  ! CHECK:       %[[V_203:[0-9]+]] = fir.if %[[V_202]] -> (f80) {
  ! CHECK:         fir.result %cst{{[_0-9]*}} : f80
  ! CHECK:       } else {
  ! CHECK:         %[[V_204:[0-9]+]] = fir.call @_FortranAMapException(%c63{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_205:[0-9]+]] = fir.call @fetestexcept(%[[V_204]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_206:[0-9]+]] = fir.call @fedisableexcept(%[[V_204]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_207:[0-9]+]] = fir.call @_FortranANearest10(%[[V_158]], %true{{[_0-9]*}}) fastmath<contract> : (f80, i1) -> f80
  ! CHECK:         %[[V_208:[0-9]+]] = fir.call @feclearexcept(%[[V_204]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_209:[0-9]+]] = fir.call @feraiseexcept(%[[V_205]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_210:[0-9]+]] = fir.call @feenableexcept(%[[V_206]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_207]] : f80
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_203]] : f80
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_167]] to %[[V_5]] : !fir.ref<f80>
  r10 = ieee_next_up(x10)
  print "('up:     ', z20.20, ' -> ', z20.20, ' = ', g0)", x10, r10, r10

  ! CHECK:     %[[V_180:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f128>
  ! CHECK:     %[[V_181:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_180]]) <{bit = 3 : i32}> : (f128) -> i1
  ! CHECK:     %[[V_182:[0-9]+]] = arith.bitcast %[[V_180]] : f128 to i128
  ! CHECK:     %[[V_183:[0-9]+]] = arith.shrui %[[V_182]], %c127{{.*}} : i128
  ! CHECK:     %[[V_184:[0-9]+]] = fir.convert %[[V_183]] : (i128) -> i1
  ! CHECK:     %[[V_185:[0-9]+]] = arith.cmpi ne, %[[V_184]], %false{{[_0-9]*}} : i1
  ! CHECK:     %[[V_186:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_180]]) <{bit = 516 : i32}> : (f128) -> i1
  ! CHECK:     %[[V_187:[0-9]+]] = arith.andi %[[V_186]], %[[V_185]] : i1
  ! CHECK:     %[[V_188:[0-9]+]] = arith.ori %[[V_181]], %[[V_187]] : i1
  ! CHECK:     %[[V_189:[0-9]+]] = fir.if %[[V_188]] -> (f128) {
  ! CHECK:       %[[V_202:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_180]]) <{bit = 1 : i32}> : (f128) -> i1
  ! CHECK:       fir.if %[[V_202]] {
  ! CHECK:         %[[V_203:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                             fir.call @feraiseexcept(%[[V_203]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_180]] : f128
  ! CHECK:     } else {
  ! CHECK:       %[[V_202:[0-9]+]] = arith.cmpf oeq, %[[V_180]], %cst{{[_0-9]*}} fastmath<contract> : f128
  ! CHECK:       %[[V_203:[0-9]+]] = fir.if %[[V_202]] -> (f128) {
  ! CHECK:         fir.result %cst{{[_0-9]*}} : f128
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_204:[0-9]+]] = arith.subi %[[V_182]], %c1{{.*}} : i128
  ! CHECK-DAG:     %[[V_205:[0-9]+]] = arith.addi %[[V_182]], %c1{{.*}} : i128
  ! CHECK:         %[[V_206:[0-9]+]] = arith.select %[[V_185]], %[[V_205]], %[[V_204]] : i128
  ! CHECK:         %[[V_207:[0-9]+]] = arith.bitcast %[[V_206]] : i128 to f128
  ! CHECK:         fir.result %[[V_207]] : f128
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_203]] : f128
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_189]] to %[[V_7]] : !fir.ref<f128>

  r16 = ieee_next_down(x16)
  print "('down:   ', z32.32, ' -> ', z32.32, ' = ', g0)", x16, r16, r16
end
