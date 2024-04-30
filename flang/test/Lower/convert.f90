! RUN: %flang_fc1 -emit-fir -fconvert=unknown %s -o - | FileCheck %s --check-prefixes=ALL,UNKNOWN
! RUN: %flang_fc1 -emit-fir -fconvert=native %s -o - | FileCheck %s --check-prefixes=ALL,NATIVE
! RUN: %flang_fc1 -emit-fir -fconvert=little-endian %s -o - | FileCheck %s --check-prefixes=ALL,LITTLE_ENDIAN
! RUN: %flang_fc1 -emit-fir -fconvert=big-endian %s -o - | FileCheck %s --check-prefixes=ALL,BIG_ENDIAN
! RUN: %flang_fc1 -emit-fir -fconvert=swap %s -o - | FileCheck %s --check-prefixes=ALL,SWAP

program test
  continue
end

! Try to test that -fconvert=<value> flag results in a environment default list
! with the FORT_CONVERT option correctly specified.

! ALL: fir.global linkonce @_QQEnvironmentDefaults.items constant : !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>> {
! ALL: %[[VAL_0:.*]] = fir.undefined !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>
! ALL: %[[VAL_1:.*]] = fir.address_of(@[[FC_STR:.*]]) : !fir.ref<!fir.char<1,13>>
! ALL: %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
! ALL: %[[VAL_4:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_3]], [0 : index, 0 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>, !fir.ref<i8>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>
! ALL: %[[VAL_5:.*]] = fir.address_of(@[[OPT_STR:.*]]) : !fir.ref<!fir.char<1,[[OPT_STR_LEN:.*]]>>
! ALL: %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,[[OPT_STR_LEN]]>>) -> !fir.ref<i8>
! ALL: %[[VAL_8:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_7]], [0 : index, 1 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>, !fir.ref<i8>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>
! ALL: fir.has_value %[[VAL_8]] : !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>

! ALL: fir.global linkonce @[[FC_STR]] constant : !fir.char<1,13> {
! ALL: %[[VAL_0:.*]] = fir.string_lit "FORT_CONVERT\00"(13) : !fir.char<1,13>
! ALL: fir.has_value %[[VAL_0]] : !fir.char<1,13>

! ALL: fir.global linkonce @[[OPT_STR]] constant : !fir.char<1,[[OPT_STR_LEN]]> {
! UNKNOWN: %[[VAL_0:.*]] = fir.string_lit "UNKNOWN\00"([[OPT_STR_LEN]]) : !fir.char<1,[[OPT_STR_LEN]]>
! NATIVE: %[[VAL_0:.*]] = fir.string_lit "NATIVE\00"([[OPT_STR_LEN]]) : !fir.char<1,[[OPT_STR_LEN]]>
! LITTLE_ENDIAN: %[[VAL_0:.*]] = fir.string_lit "LITTLE_ENDIAN\00"([[OPT_STR_LEN]]) : !fir.char<1,[[OPT_STR_LEN]]>
! BIG_ENDIAN: %[[VAL_0:.*]] = fir.string_lit "BIG_ENDIAN\00"([[OPT_STR_LEN]]) : !fir.char<1,[[OPT_STR_LEN]]>
! SWAP: %[[VAL_0:.*]] = fir.string_lit "SWAP\00"([[OPT_STR_LEN]]) : !fir.char<1,[[OPT_STR_LEN]]>
! ALL: fir.has_value %[[VAL_0]] : !fir.char<1,[[OPT_STR_LEN]]>

! ALL: fir.global linkonce @_QQEnvironmentDefaults.list constant : tuple<i[[int_size:.*]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>> {
! ALL: %[[VAL_1:.*]] = arith.constant 1 : i[[int_size]]
! ALL: %[[VAL_0:.*]] = fir.undefined tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>
! ALL: %[[VAL_2:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_1]], [0 : index] : (tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>, i[[int_size]]) -> tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>
! ALL: %[[VAL_3:.*]] = fir.address_of(@_QQEnvironmentDefaults.items) : !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>
! ALL: %[[VAL_4:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_3]], [1 : index] : (tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>) -> tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>
! ALL: fir.has_value %[[VAL_4]] : tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>

! ALL: fir.global @_QQEnvironmentDefaults constant : !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>> {
! ALL: %[[VAL_0:.*]] = fir.address_of(@_QQEnvironmentDefaults.list) : !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
! ALL: fir.has_value %[[VAL_0]] : !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
