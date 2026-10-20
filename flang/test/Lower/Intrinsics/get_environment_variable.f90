! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 -DDEFAULT_LOGICAL_SIZE=4 %s
! RUN: %flang_fc1 -fdefault-integer-8 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 -DDEFAULT_LOGICAL_SIZE=8 %s

! CHECK-LABEL: func @_QPname_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"}) {
subroutine name_only(name)
    character(len=32) :: name
    call get_environment_variable(name)
! CHECK-NOT: fir.call @_FortranAGetEnvVariable
! CHECK: return
end subroutine name_only

! CHECK-LABEL: func @_QPname_and_value_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"}) {
subroutine name_and_value_only(name, value)
    character(len=32) :: name, value
    call get_environment_variable(name, value)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[valueCast:.*]] = fir.convert %[[valueUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[valueCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[valueBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %true = arith.constant true
! CHECK: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[value:.*]] = fir.convert %[[valueBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: return
end subroutine name_and_value_only

! CHECK-LABEL: func @_QPname_and_length_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[lengthArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"}) {
subroutine name_and_length_only(name, length)
    character(len=32) :: name
    integer :: length
    call get_environment_variable(name, LENGTH=length)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[lengthBox:.*]] = fir.embox {{.*}} : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK: %true = arith.constant true
! CHECK: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[length:.*]] = fir.convert %[[lengthBox]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine name_and_length_only

! CHECK-LABEL: func @_QPname_and_status_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"}) {
subroutine name_and_status_only(name, status)
    character(len=32) :: name
    integer :: status
    call get_environment_variable(name, STATUS=status)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %true = arith.constant true
! CHECK: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32: %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status32:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to {{.*}} : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine name_and_status_only

! CHECK-LABEL: func @_QPname_and_trim_name_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>> {fir.bindc_name = "trim_name"}) {
subroutine name_and_trim_name_only(name, trim_name)
    character(len=32) :: name
    logical :: trim_name
    call get_environment_variable(name, TRIM_NAME=trim_name)
! CHECK-NOT: fir.call @_FortranAGetEnvVariable
! CHECK: return
end subroutine name_and_trim_name_only

! CHECK-LABEL: func @_QPname_and_errmsg_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine name_and_errmsg_only(name, errmsg)
    character(len=32) :: name, errmsg
    call get_environment_variable(name, ERRMSG=errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[errmsgCast]]
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[errmsgBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %true = arith.constant true
! CHECK: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.convert %[[errmsgBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: return
end subroutine name_and_errmsg_only

! CHECK-LABEL: func @_QPall_arguments(
! CHECK-SAME: %[[nameArg:[^:]*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"},
! CHECK-SAME: %[[lengthArg:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"},
! CHECK-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>> {fir.bindc_name = "trim_name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine all_arguments(name, value, length, status, trim_name, errmsg)
    character(len=32) :: name, value, errmsg
    integer :: length, status
    logical :: trim_name
    call get_environment_variable(name, value, length, status, trim_name, errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[errmsgCast]]
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[valueCast:.*]] = fir.convert %[[valueUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[valueCast]]
! CHECK: %[[nameBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[valueBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[lengthBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK: %[[errmsgBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK:      %[[trimName:.*]] = fir.if %{{.*}} -> (i1) {
! CHECK:   %[[trimNameLoaded:.*]] = fir.load {{.*}} : !fir.ref<!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>>
! CHECK:   %[[trimCast:.*]] = fir.convert %[[trimNameLoaded]] : (!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>) -> i1
! CHECK:   fir.result %[[trimCast]] : i1
! CHECK: } else {
! CHECK:   %[[trueVal:.*]] = arith.constant true
! CHECK:   fir.result %[[trueVal]] : i1
! CHECK: }
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX[[fileString:.*]]) : !fir.ref<!fir.char<1,[[fileStringLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[value:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[length:.*]] = fir.convert %[[lengthBoxed]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[fileStringLength]]>>) -> !fir.ref<i8>
! CHECK-32: %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status32:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to {{.*}} : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine all_arguments


! CHECK-LABEL: func @_QPgetenv_name_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"}) {
subroutine getenv_name_only(name)
    character(len=32) :: name
    call getenv(name)
! CHECK-NOT: fir.call @_FortranAGetEnvVariable
! CHECK: return
end subroutine getenv_name_only

! CHECK-LABEL: func @_QPgetenv_name_and_value_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"}) {
subroutine getenv_name_and_value_only(name, value)
    character(len=32) :: name, value
    call getenv(name, value)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[valueCast:.*]] = fir.convert %[[valueUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[valueCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[valueBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %true = arith.constant true
! CHECK: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[value:.*]] = fir.convert %[[valueBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: return
end subroutine getenv_name_and_value_only

! CHECK-LABEL: func @_QPgetenv_name_and_length_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[lengthArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"}) {
subroutine getenv_name_and_length_only(name, length)
    character(len=32) :: name
    integer :: length
    call getenv(name, LENGTH=length)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[lengthBox:.*]] = fir.embox {{.*}} : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK: %true = arith.constant true
! CHECK: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[length:.*]] = fir.convert %[[lengthBox]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine getenv_name_and_length_only

! CHECK-LABEL: func @_QPgetenv_name_and_status_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"}) {
subroutine getenv_name_and_status_only(name, status)
    character(len=32) :: name
    integer :: status
    call getenv(name, STATUS=status)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %true = arith.constant true
! CHECK: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32: %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status32:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to {{.*}} : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine getenv_name_and_status_only

! CHECK-LABEL: func @_QPgetenv_name_and_trim_name_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>> {fir.bindc_name = "trim_name"}) {
subroutine getenv_name_and_trim_name_only(name, trim_name)
    character(len=32) :: name
    logical :: trim_name
    call getenv(name, TRIM_NAME=trim_name)
! CHECK-NOT: fir.call @_FortranAGetEnvVariable
! CHECK: return
end subroutine getenv_name_and_trim_name_only

! CHECK-LABEL: func @_QPgetenv_name_and_errmsg_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine getenv_name_and_errmsg_only(name, errmsg)
    character(len=32) :: name, errmsg
    call getenv(name, ERRMSG=errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[errmsgCast]]
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[nameBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[errmsgBox:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %true = arith.constant true
! CHECK: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.convert %[[errmsgBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: return
end subroutine getenv_name_and_errmsg_only

! CHECK-LABEL: func @_QPgetenv_all_arguments(
! CHECK-SAME: %[[nameArg:[^:]*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"},
! CHECK-SAME: %[[lengthArg:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"},
! CHECK-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>> {fir.bindc_name = "trim_name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine getenv_all_arguments(name, value, length, status, trim_name, errmsg)
    character(len=32) :: name, value, errmsg
    integer :: length, status
    logical :: trim_name
    call getenv(name, value, length, status, trim_name, errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[errmsgCast]]
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[nameCast]]
! CHECK: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[valueCast:.*]] = fir.convert %[[valueUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK: hlfir.declare %[[valueCast]]
! CHECK: %[[nameBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[valueBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[lengthBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK: %[[errmsgBoxed:.*]] = fir.embox {{.*}} : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK:      %[[trimName:.*]] = fir.if %{{.*}} -> (i1) {
! CHECK:   %[[trimNameLoaded:.*]] = fir.load {{.*}} : !fir.ref<!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>>
! CHECK:   %[[trimCast:.*]] = fir.convert %[[trimNameLoaded]] : (!fir.logical<[[DEFAULT_LOGICAL_SIZE]]>) -> i1
! CHECK:   fir.result %[[trimCast]] : i1
! CHECK: } else {
! CHECK:   %[[trueVal:.*]] = arith.constant true
! CHECK:   fir.result %[[trueVal]] : i1
! CHECK: }
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQclX[[fileString:.*]]) : !fir.ref<!fir.char<1,[[fileStringLength:.*]]>>
! CHECK: %[[sourceLine:.*]] = arith.constant [[#]] : i32
! CHECK: %[[name:.*]] = fir.convert %[[nameBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[value:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[length:.*]] = fir.convert %[[lengthBoxed]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[fileStringLength]]>>) -> !fir.ref<i8>
! CHECK-32: %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status32:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to {{.*}} : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine getenv_all_arguments
