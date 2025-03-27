! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: system_clock_test
subroutine system_clock_test()
  integer(4) :: c
  integer(8) :: m
  real :: r
  ! CHECK-DAG: %[[c:.*]] = fir.alloca i32 {bindc_name = "c"
  ! CHECK-DAG: %[[m:.*]] = fir.alloca i64 {bindc_name = "m"
  ! CHECK-DAG: %[[r:.*]] = fir.alloca f32 {bindc_name = "r"
  ! CHECK: %[[c4:.*]] = arith.constant 4 : i32
  ! CHECK: %[[Count:.*]] = fir.call @_FortranASystemClockCount(%[[c4]]) {{.*}}: (i32) -> i64
  ! CHECK: %[[Count1:.*]] = fir.convert %[[Count]] : (i64) -> i32
  ! CHECK: fir.store %[[Count1]] to %[[c]] : !fir.ref<i32>
  ! CHECK: %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8]]) {{.*}}: (i32) -> i64
  ! CHECK: %[[Rate1:.*]] = fir.convert %[[Rate]] : (i64) -> f32
  ! CHECK: fir.store %[[Rate1]] to %[[r]] : !fir.ref<f32>
  ! CHECK: %[[c8_2:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Max:.*]] = fir.call @_FortranASystemClockCountMax(%[[c8_2]]) {{.*}}: (i32) -> i64
  ! CHECK: fir.store %[[Max]] to %[[m]] : !fir.ref<i64>
  call system_clock(c, r, m)
! print*, c, r, m
  ! CHECK-NOT: fir.call
  ! CHECK: %[[c8_3:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8_3]]) {{.*}}: (i32) -> i64
  ! CHECK: fir.store %[[Rate]] to %[[m]] : !fir.ref<i64>
  call system_clock(count_rate=m)
  ! CHECK-NOT: fir.call
! print*, m
end subroutine

subroutine ss(count)
! CHECK-LABEL:   func.func @_QPss(
! CHECK-SAME:                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i64> {fir.bindc_name = "count", fir.optional}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<i64>> {bindc_name = "count_max", uniq_name = "_QFssEcount_max"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.heap<i64> {uniq_name = "_QFssEcount_max.addr"}
! CHECK:           %[[VAL_3:.*]] = fir.zero_bits !fir.heap<i64>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<i64>> {bindc_name = "count_rate", uniq_name = "_QFssEcount_rate"}
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.ptr<i64> {uniq_name = "_QFssEcount_rate.addr"}
! CHECK:           %[[VAL_6:.*]] = fir.zero_bits !fir.ptr<i64>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_7:.*]] = fir.alloca i64 {bindc_name = "count_rate_", fir.target, uniq_name = "_QFssEcount_rate_"}
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i64>) -> !fir.ptr<i64>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_9:.*]] = arith.constant false
! CHECK:           %[[VAL_10:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QQclX0146bbb9ee5e88a6e67c6c1cf8871123) : !fir.ref<!fir.char<1,76>>
! CHECK:           %[[VAL_12:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_13]] : (!fir.heap<i64>) -> !fir.box<!fir.heap<i64>>
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<i64>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_15]], %[[VAL_9]], %[[VAL_10]], %[[VAL_16]], %[[VAL_12]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_22:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i64>) -> i1
! CHECK:           fir.if %[[VAL_22]] {
! CHECK:             %[[VAL_23:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_24:.*]] = fir.call @_FortranASystemClockCount(%[[VAL_23]]) fastmath<contract> : (i32) -> i64
! CHECK:             fir.store %[[VAL_24]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:           }
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_20]] : (!fir.ptr<i64>) -> i64
! CHECK:           %[[VAL_26:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_27:.*]] = arith.cmpi ne, %[[VAL_25]], %[[VAL_26]] : i64
! CHECK:           fir.if %[[VAL_27]] {
! CHECK:             %[[VAL_28:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_29:.*]] = fir.call @_FortranASystemClockCountRate(%[[VAL_28]]) fastmath<contract> : (i32) -> i64
! CHECK:             fir.store %[[VAL_29]] to %[[VAL_20]] : !fir.ptr<i64>
! CHECK:           }
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_21]] : (!fir.heap<i64>) -> i64
! CHECK:           %[[VAL_31:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_31]] : i64
! CHECK:           fir.if %[[VAL_32]] {
! CHECK:             %[[VAL_33:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_34:.*]] = fir.call @_FortranASystemClockCountMax(%[[VAL_33]]) fastmath<contract> : (i32) -> i64
! CHECK:             fir.store %[[VAL_34]] to %[[VAL_21]] : !fir.heap<i64>
! CHECK:           }
! CHECK:           %[[VAL_35:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i64>) -> i1
! CHECK:           fir.if %[[VAL_35]] {
! CHECK:             %[[VAL_36:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_37:.*]] = fir.address_of(@_QQclX0146bbb9ee5e88a6e67c6c1cf8871123) : !fir.ref<!fir.char<1,76>>
! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_39:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_40:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_36]], %[[VAL_38]], %[[VAL_39]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:             %[[VAL_42:.*]] = fir.call @_FortranAioOutputInteger64(%[[VAL_40]], %[[VAL_41]]) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ptr<i64>
! CHECK:             %[[VAL_45:.*]] = fir.call @_FortranAioOutputInteger64(%[[VAL_40]], %[[VAL_44]]) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
! CHECK:             %[[VAL_46:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:             %[[VAL_47:.*]] = fir.load %[[VAL_46]] : !fir.heap<i64>
! CHECK:             %[[VAL_48:.*]] = fir.call @_FortranAioOutputInteger64(%[[VAL_40]], %[[VAL_47]]) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
! CHECK:             %[[VAL_49:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_40]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           } else {
! CHECK:             %[[VAL_50:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:             %[[VAL_51:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_50]] : (!fir.ptr<i64>) -> i64
! CHECK:             %[[VAL_53:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_54:.*]] = arith.cmpi ne, %[[VAL_52]], %[[VAL_53]] : i64
! CHECK:             fir.if %[[VAL_54]] {
! CHECK:               %[[VAL_55:.*]] = arith.constant {{.*}} : i32
! CHECK:               %[[VAL_56:.*]] = fir.call @_FortranASystemClockCountRate(%[[VAL_55]]) fastmath<contract> : (i32) -> i64
! CHECK:               fir.store %[[VAL_56]] to %[[VAL_50]] : !fir.ptr<i64>
! CHECK:             }
! CHECK:             %[[VAL_57:.*]] = fir.convert %[[VAL_51]] : (!fir.heap<i64>) -> i64
! CHECK:             %[[VAL_58:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_59:.*]] = arith.cmpi ne, %[[VAL_57]], %[[VAL_58]] : i64
! CHECK:             fir.if %[[VAL_59]] {
! CHECK:               %[[VAL_60:.*]] = arith.constant {{.*}} : i32
! CHECK:               %[[VAL_61:.*]] = fir.call @_FortranASystemClockCountMax(%[[VAL_60]]) fastmath<contract> : (i32) -> i64
! CHECK:               fir.store %[[VAL_61]] to %[[VAL_51]] : !fir.heap<i64>
! CHECK:             }
! CHECK:             %[[VAL_62:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_63:.*]] = fir.address_of(@_QQclX0146bbb9ee5e88a6e67c6c1cf8871123) : !fir.ref<!fir.char<1,76>>
! CHECK:             %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_65:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_66:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_62]], %[[VAL_64]], %[[VAL_65]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:             %[[VAL_67:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:             %[[VAL_68:.*]] = fir.load %[[VAL_67]] : !fir.ptr<i64>
! CHECK:             %[[VAL_69:.*]] = fir.call @_FortranAioOutputInteger64(%[[VAL_66]], %[[VAL_68]]) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
! CHECK:             %[[VAL_70:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:             %[[VAL_71:.*]] = fir.load %[[VAL_70]] : !fir.heap<i64>
! CHECK:             %[[VAL_72:.*]] = fir.call @_FortranAioOutputInteger64(%[[VAL_66]], %[[VAL_71]]) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
! CHECK:             %[[VAL_73:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_66]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           }
! CHECK:           %[[VAL_74:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i64>) -> i1
! CHECK:           fir.if %[[VAL_74]] {
! CHECK:             %[[VAL_75:.*]] = arith.constant 0 : i64
! CHECK:             fir.store %[[VAL_75]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:           }
! CHECK:           %[[VAL_76:.*]] = fir.zero_bits !fir.ptr<i64>
! CHECK:           fir.store %[[VAL_76]] to %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_77:.*]] = arith.constant false
! CHECK:           %[[VAL_78:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_79:.*]] = fir.address_of(@_QQclX0146bbb9ee5e88a6e67c6c1cf8871123) : !fir.ref<!fir.char<1,76>>
! CHECK:           %[[VAL_80:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[VAL_81:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_82:.*]] = fir.embox %[[VAL_81]] : (!fir.heap<i64>) -> !fir.box<!fir.heap<i64>>
! CHECK:           fir.store %[[VAL_82]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:           %[[VAL_83:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<i64>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_84:.*]] = fir.convert %[[VAL_79]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_85:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_83]], %[[VAL_77]], %[[VAL_78]], %[[VAL_84]], %[[VAL_80]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_86:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:           %[[VAL_87:.*]] = fir.box_addr %[[VAL_86]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:           fir.store %[[VAL_87]] to %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_88:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_89:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_90:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i64>) -> i1
! CHECK:           fir.if %[[VAL_90]] {
! CHECK:             %[[VAL_91:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_92:.*]] = fir.call @_FortranASystemClockCount(%[[VAL_91]]) fastmath<contract> : (i32) -> i64
! CHECK:             fir.store %[[VAL_92]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:           }
! CHECK:           %[[VAL_93:.*]] = fir.convert %[[VAL_88]] : (!fir.ptr<i64>) -> i64
! CHECK:           %[[VAL_94:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_95:.*]] = arith.cmpi ne, %[[VAL_93]], %[[VAL_94]] : i64
! CHECK:           fir.if %[[VAL_95]] {
! CHECK:             %[[VAL_96:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_97:.*]] = fir.call @_FortranASystemClockCountRate(%[[VAL_96]]) fastmath<contract> : (i32) -> i64
! CHECK:             fir.store %[[VAL_97]] to %[[VAL_88]] : !fir.ptr<i64>
! CHECK:           }
! CHECK:           %[[VAL_98:.*]] = fir.convert %[[VAL_89]] : (!fir.heap<i64>) -> i64
! CHECK:           %[[VAL_99:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_100:.*]] = arith.cmpi ne, %[[VAL_98]], %[[VAL_99]] : i64
! CHECK:           fir.if %[[VAL_100]] {
! CHECK:             %[[VAL_101:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_102:.*]] = fir.call @_FortranASystemClockCountMax(%[[VAL_101]]) fastmath<contract> : (i32) -> i64
! CHECK:             fir.store %[[VAL_102]] to %[[VAL_89]] : !fir.heap<i64>
! CHECK:           }
! CHECK:           %[[VAL_103:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i64>) -> i1
! CHECK:           fir.if %[[VAL_103]] {
! CHECK:             %[[VAL_104:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_105:.*]] = fir.address_of(@_QQclX0146bbb9ee5e88a6e67c6c1cf8871123) : !fir.ref<!fir.char<1,76>>
! CHECK:             %[[VAL_106:.*]] = fir.convert %[[VAL_105]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_107:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_108:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_104]], %[[VAL_106]], %[[VAL_107]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:             %[[VAL_109:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:             %[[VAL_110:.*]] = fir.call @_FortranAioOutputInteger64(%[[VAL_108]], %[[VAL_109]]) fastmath<contract> : (!fir.ref<i8>, i64) -> i1
! CHECK:             %[[VAL_111:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_108]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           }
! CHECK:           %[[VAL_112:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           %[[VAL_113:.*]] = fir.convert %[[VAL_112]] : (!fir.heap<i64>) -> i64
! CHECK:           %[[VAL_114:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_115:.*]] = arith.cmpi ne, %[[VAL_113]], %[[VAL_114]] : i64
! CHECK:           fir.if %[[VAL_115]] {
! CHECK:             %[[VAL_116:.*]] = arith.constant false
! CHECK:             %[[VAL_117:.*]] = fir.absent !fir.box<none>
! CHECK:             %[[VAL_118:.*]] = fir.address_of(@_QQclX0146bbb9ee5e88a6e67c6c1cf8871123) : !fir.ref<!fir.char<1,76>>
! CHECK:             %[[VAL_119:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_120:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:             %[[VAL_121:.*]] = fir.embox %[[VAL_120]] : (!fir.heap<i64>) -> !fir.box<!fir.heap<i64>>
! CHECK:             fir.store %[[VAL_121]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:             %[[VAL_122:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<i64>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_123:.*]] = fir.convert %[[VAL_118]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_124:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_122]], %[[VAL_116]], %[[VAL_117]], %[[VAL_123]], %[[VAL_119]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:             %[[VAL_125:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:             %[[VAL_126:.*]] = fir.box_addr %[[VAL_125]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:             fir.store %[[VAL_126]] to %[[VAL_2]] : !fir.ref<!fir.heap<i64>>
! CHECK:           }
! CHECK:           return
! CHECK:         }

  integer(8), optional :: count
  integer(8), target :: count_rate_
  integer(8), pointer :: count_rate
  integer(8), allocatable :: count_max

  count_rate => count_rate_
  allocate(count_max)
  call system_clock(count, count_rate, count_max)
  if (present(count)) then
    print*, count, count_rate, count_max
  else
    call system_clock(count_rate=count_rate, count_max=count_max)
    print*, count_rate, count_max
  endif

  if (present(count)) count = 0
  count_rate => null()
  deallocate(count_max)
  call system_clock(count, count_rate, count_max)
  if (present(count)) print*, count
end
