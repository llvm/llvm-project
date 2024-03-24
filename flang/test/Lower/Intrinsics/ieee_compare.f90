! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program p
  use ieee_arithmetic

  ! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  ! CHECK:     %[[V_1:[0-9]+]] = fir.declare %[[V_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK:     %[[V_58:[0-9]+]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
  ! CHECK:     %[[V_59:[0-9]+]] = fir.declare %[[V_58]] {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK:     %[[V_60:[0-9]+]] = fir.address_of(@_QFEx) : !fir.ref<!fir.array<10xf32>>
  ! CHECK:     %[[V_61:[0-9]+]] = fir.shape %c10{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_62:[0-9]+]] = fir.declare %[[V_60]](%[[V_61]]) {uniq_name = "_QFEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  real(4) :: x(10)

  x( 1) = ieee_value(x(1), ieee_signaling_nan)
  x( 2) = ieee_value(x(1), ieee_quiet_nan)
  x( 3) = ieee_value(x(1), ieee_negative_inf)
  x( 4) = ieee_value(x(1), ieee_negative_normal)
  x( 5) = ieee_value(x(1), ieee_negative_denormal)
  x( 6) = ieee_value(x(1), ieee_negative_zero)
  x( 7) = ieee_value(x(1), ieee_positive_zero)
  x( 8) = ieee_value(x(1), ieee_positive_denormal)
  x( 9) = ieee_value(x(1), ieee_positive_normal)
  x(10) = ieee_value(x(1), ieee_positive_inf)

  do i = lbound(x,1), ubound(x,1)
    do j = lbound(x,1), ubound(x,1)
      ! CHECK:     %[[V_153:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_174:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_175:[0-9]+]] = fir.convert %[[V_174]] : (i32) -> i64
      ! CHECK:     %[[V_176:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_175]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_177:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_178:[0-9]+]] = fir.convert %[[V_177]] : (i32) -> i64
      ! CHECK:     %[[V_179:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_178]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_180:[0-9]+]] = fir.load %[[V_176]] : !fir.ref<f32>
      ! CHECK:     %[[V_181:[0-9]+]] = fir.load %[[V_179]] : !fir.ref<f32>
      ! CHECK-DAG: %[[V_182:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_181]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG: %[[V_183:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_180]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:     %[[V_184:[0-9]+]] = arith.ori %[[V_183]], %[[V_182]] : i1
      ! CHECK:     %[[V_185:[0-9]+]] = arith.cmpf oeq, %[[V_180]], %[[V_181]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_184]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_186:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_153]], %[[V_185]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [Q]', x(i), 'eq', x(j), ieee_quiet_eq(x(i), x(j))

      ! CHECK:     %[[V_188:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_206:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_207:[0-9]+]] = fir.convert %[[V_206]] : (i32) -> i64
      ! CHECK:     %[[V_208:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_207]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_209:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_210:[0-9]+]] = fir.convert %[[V_209]] : (i32) -> i64
      ! CHECK:     %[[V_211:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_210]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_212:[0-9]+]] = fir.load %[[V_208]] : !fir.ref<f32>
      ! CHECK:     %[[V_213:[0-9]+]] = fir.load %[[V_211]] : !fir.ref<f32>
      ! CHECK-DAG: %[[V_214:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_213]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG: %[[V_215:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_212]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:     %[[V_216:[0-9]+]] = arith.ori %[[V_215]], %[[V_214]] : i1
      ! CHECK:     %[[V_217:[0-9]+]] = arith.cmpf oge, %[[V_212]], %[[V_213]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_216]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_218:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_188]], %[[V_217]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [Q]', x(i), 'ge', x(j), ieee_quiet_ge(x(i), x(j))

      ! CHECK:     %[[V_220:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_238:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_239:[0-9]+]] = fir.convert %[[V_238]] : (i32) -> i64
      ! CHECK:     %[[V_240:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_239]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_241:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_242:[0-9]+]] = fir.convert %[[V_241]] : (i32) -> i64
      ! CHECK:     %[[V_243:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_242]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_244:[0-9]+]] = fir.load %[[V_240]] : !fir.ref<f32>
      ! CHECK:     %[[V_245:[0-9]+]] = fir.load %[[V_243]] : !fir.ref<f32>
      ! CHECK-DAG: %[[V_246:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_245]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG: %[[V_247:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_244]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:     %[[V_248:[0-9]+]] = arith.ori %[[V_247]], %[[V_246]] : i1
      ! CHECK:     %[[V_249:[0-9]+]] = arith.cmpf ogt, %[[V_244]], %[[V_245]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_248]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_250:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_220]], %[[V_249]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [Q]', x(i), 'gt', x(j), ieee_quiet_gt(x(i), x(j))

      ! CHECK:     %[[V_252:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_270:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_271:[0-9]+]] = fir.convert %[[V_270]] : (i32) -> i64
      ! CHECK:     %[[V_272:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_271]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_273:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_274:[0-9]+]] = fir.convert %[[V_273]] : (i32) -> i64
      ! CHECK:     %[[V_275:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_274]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_276:[0-9]+]] = fir.load %[[V_272]] : !fir.ref<f32>
      ! CHECK:     %[[V_277:[0-9]+]] = fir.load %[[V_275]] : !fir.ref<f32>
      ! CHECK-DAG: %[[V_278:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_277]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG: %[[V_279:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_276]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:     %[[V_280:[0-9]+]] = arith.ori %[[V_279]], %[[V_278]] : i1
      ! CHECK:     %[[V_281:[0-9]+]] = arith.cmpf ole, %[[V_276]], %[[V_277]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_280]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_282:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_252]], %[[V_281]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [Q]', x(i), 'le', x(j), ieee_quiet_le(x(i), x(j))

      ! CHECK:     %[[V_284:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_302:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_303:[0-9]+]] = fir.convert %[[V_302]] : (i32) -> i64
      ! CHECK:     %[[V_304:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_303]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_305:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_306:[0-9]+]] = fir.convert %[[V_305]] : (i32) -> i64
      ! CHECK:     %[[V_307:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_306]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_308:[0-9]+]] = fir.load %[[V_304]] : !fir.ref<f32>
      ! CHECK:     %[[V_309:[0-9]+]] = fir.load %[[V_307]] : !fir.ref<f32>
      ! CHECK-DAG: %[[V_310:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_309]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG: %[[V_311:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_308]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:     %[[V_312:[0-9]+]] = arith.ori %[[V_311]], %[[V_310]] : i1
      ! CHECK:     %[[V_313:[0-9]+]] = arith.cmpf olt, %[[V_308]], %[[V_309]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_312]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_314:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_284]], %[[V_313]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [Q]', x(i), 'lt', x(j), ieee_quiet_lt(x(i), x(j))

      ! CHECK:     %[[V_316:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_334:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_335:[0-9]+]] = fir.convert %[[V_334]] : (i32) -> i64
      ! CHECK:     %[[V_336:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_335]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_337:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_338:[0-9]+]] = fir.convert %[[V_337]] : (i32) -> i64
      ! CHECK:     %[[V_339:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_338]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_340:[0-9]+]] = fir.load %[[V_336]] : !fir.ref<f32>
      ! CHECK:     %[[V_341:[0-9]+]] = fir.load %[[V_339]] : !fir.ref<f32>
      ! CHECK-DAG: %[[V_342:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_341]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG: %[[V_343:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_340]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:     %[[V_344:[0-9]+]] = arith.ori %[[V_343]], %[[V_342]] : i1
      ! CHECK:     %[[V_345:[0-9]+]] = arith.cmpf une, %[[V_340]], %[[V_341]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_344]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_346:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_316]], %[[V_345]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [Q]', x(i), 'ne', x(j), ieee_quiet_ne(x(i), x(j))

      ! CHECK:     %[[V_348:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_366:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_367:[0-9]+]] = fir.convert %[[V_366]] : (i32) -> i64
      ! CHECK:     %[[V_368:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_367]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_369:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_370:[0-9]+]] = fir.convert %[[V_369]] : (i32) -> i64
      ! CHECK:     %[[V_371:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_370]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_372:[0-9]+]] = fir.load %[[V_368]] : !fir.ref<f32>
      ! CHECK:     %[[V_373:[0-9]+]] = fir.load %[[V_371]] : !fir.ref<f32>
      ! CHECK:     %[[V_374:[0-9]+]] = arith.cmpf uno, %[[V_372]], %[[V_373]] {{.*}} : f32
      ! CHECK:     %[[V_375:[0-9]+]] = arith.cmpf oeq, %[[V_372]], %[[V_373]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_374]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_376:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_348]], %[[V_375]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [S]', x(i), 'eq', x(j), ieee_signaling_eq(x(i), x(j))

      ! CHECK:     %[[V_378:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_395:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_396:[0-9]+]] = fir.convert %[[V_395]] : (i32) -> i64
      ! CHECK:     %[[V_397:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_396]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_398:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_399:[0-9]+]] = fir.convert %[[V_398]] : (i32) -> i64
      ! CHECK:     %[[V_400:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_399]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_401:[0-9]+]] = fir.load %[[V_397]] : !fir.ref<f32>
      ! CHECK:     %[[V_402:[0-9]+]] = fir.load %[[V_400]] : !fir.ref<f32>
      ! CHECK:     %[[V_403:[0-9]+]] = arith.cmpf uno, %[[V_401]], %[[V_402]] {{.*}} : f32
      ! CHECK:     %[[V_404:[0-9]+]] = arith.cmpf oge, %[[V_401]], %[[V_402]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_403]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_405:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_378]], %[[V_404]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [S]', x(i), 'ge', x(j), ieee_signaling_ge(x(i), x(j))

      ! CHECK:     %[[V_407:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_424:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_425:[0-9]+]] = fir.convert %[[V_424]] : (i32) -> i64
      ! CHECK:     %[[V_426:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_425]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_427:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_428:[0-9]+]] = fir.convert %[[V_427]] : (i32) -> i64
      ! CHECK:     %[[V_429:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_428]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_430:[0-9]+]] = fir.load %[[V_426]] : !fir.ref<f32>
      ! CHECK:     %[[V_431:[0-9]+]] = fir.load %[[V_429]] : !fir.ref<f32>
      ! CHECK:     %[[V_432:[0-9]+]] = arith.cmpf uno, %[[V_430]], %[[V_431]] {{.*}} : f32
      ! CHECK:     %[[V_433:[0-9]+]] = arith.cmpf ogt, %[[V_430]], %[[V_431]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_432]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_434:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_407]], %[[V_433]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [S]', x(i), 'gt', x(j), ieee_signaling_gt(x(i), x(j))

      ! CHECK:     %[[V_436:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_453:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_454:[0-9]+]] = fir.convert %[[V_453]] : (i32) -> i64
      ! CHECK:     %[[V_455:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_454]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_456:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_457:[0-9]+]] = fir.convert %[[V_456]] : (i32) -> i64
      ! CHECK:     %[[V_458:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_457]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_459:[0-9]+]] = fir.load %[[V_455]] : !fir.ref<f32>
      ! CHECK:     %[[V_460:[0-9]+]] = fir.load %[[V_458]] : !fir.ref<f32>
      ! CHECK:     %[[V_461:[0-9]+]] = arith.cmpf uno, %[[V_459]], %[[V_460]] {{.*}} : f32
      ! CHECK:     %[[V_462:[0-9]+]] = arith.cmpf ole, %[[V_459]], %[[V_460]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_461]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_463:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_436]], %[[V_462]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [S]', x(i), 'le', x(j), ieee_signaling_le(x(i), x(j))

      ! CHECK:     %[[V_465:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_482:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_483:[0-9]+]] = fir.convert %[[V_482]] : (i32) -> i64
      ! CHECK:     %[[V_484:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_483]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_485:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_486:[0-9]+]] = fir.convert %[[V_485]] : (i32) -> i64
      ! CHECK:     %[[V_487:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_486]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_488:[0-9]+]] = fir.load %[[V_484]] : !fir.ref<f32>
      ! CHECK:     %[[V_489:[0-9]+]] = fir.load %[[V_487]] : !fir.ref<f32>
      ! CHECK:     %[[V_490:[0-9]+]] = arith.cmpf uno, %[[V_488]], %[[V_489]] {{.*}} : f32
      ! CHECK:     %[[V_491:[0-9]+]] = arith.cmpf olt, %[[V_488]], %[[V_489]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_490]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_492:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_465]], %[[V_491]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [S]', x(i), 'lt', x(j), ieee_signaling_lt(x(i), x(j))

      ! CHECK:     %[[V_494:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
      ! CHECK:                         fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_511:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
      ! CHECK:     %[[V_512:[0-9]+]] = fir.convert %[[V_511]] : (i32) -> i64
      ! CHECK:     %[[V_513:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_512]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_514:[0-9]+]] = fir.load %[[V_59]] : !fir.ref<i32>
      ! CHECK:     %[[V_515:[0-9]+]] = fir.convert %[[V_514]] : (i32) -> i64
      ! CHECK:     %[[V_516:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_61]]) %[[V_515]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      ! CHECK:     %[[V_517:[0-9]+]] = fir.load %[[V_513]] : !fir.ref<f32>
      ! CHECK:     %[[V_518:[0-9]+]] = fir.load %[[V_516]] : !fir.ref<f32>
      ! CHECK:     %[[V_519:[0-9]+]] = arith.cmpf uno, %[[V_517]], %[[V_518]] {{.*}} : f32
      ! CHECK:     %[[V_520:[0-9]+]] = arith.cmpf une, %[[V_517]], %[[V_518]] {{.*}} : f32
      ! CHECK:     fir.if %[[V_519]] {
      ! CHECK:       %[[V_526:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:       %[[V_527:[0-9]+]] = fir.call @feraiseexcept(%[[V_526]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_521:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_494]], %[[V_520]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
      print*, ' [S]', x(i), 'ne', x(j), ieee_signaling_ne(x(i), x(j))
    enddo
  enddo
end
