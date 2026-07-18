! RUN: %flang_fc1 -emit-hlfir -freal-sum-reassociation -o - %s | FileCheck %s --check-prefixes=SPLIT,NO-REWRITE
! RUN: %flang_fc1 -emit-hlfir -fno-real-sum-reassociation -o - %s | FileCheck %s --check-prefixes=DEFAULT,NO-REWRITE
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s --check-prefixes=DEFAULT,NO-REWRITE

! Default:   (((x + a*b) + c*d) + e*f)
! Rewritten: ((c*d + e*f) + (x + a*b))
subroutine eligible_self_update3(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x + a*b + c*d + e*f
end

! SPLIT-LABEL: func.func @_QPeligible_self_update3
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ea"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Eb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ec"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ed"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ee"}
! SPLIT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ef"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ex"}
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! SPLIT: %[[EV:.*]] = fir.load %[[E]]#0
! SPLIT: %[[FV:.*]] = fir.load %[[F]]#0
! SPLIT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! SPLIT: %[[TAIL:.*]] = arith.addf %[[CD]], %[[EF]]
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AB]]
! SPLIT-NOT: arith.addf %[[HEAD]], %[[CD]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_self_update3
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ea"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Eb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ec"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ed"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ee"}
! DEFAULT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ef"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ex"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! DEFAULT: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! DEFAULT: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! DEFAULT: %[[EV:.*]] = fir.load %[[E]]#0
! DEFAULT: %[[FV:.*]] = fir.load %[[F]]#0
! DEFAULT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! Default:   ((((x + a*b) + c*d) + e*f) + g*h)
! Rewritten: ((c*d + (e*f + g*h)) + (x + a*b))
subroutine eligible_self_update4(x,a,b,c,d,e,f,g,h)
  real(8) :: x,a,b,c,d,e,f,g,h
  x = x + a*b + c*d + e*f + g*h
end

! SPLIT-LABEL: func.func @_QPeligible_self_update4
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ea"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Eb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ec"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ed"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ee"}
! SPLIT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ef"}
! SPLIT-DAG: %[[G:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Eg"}
! SPLIT-DAG: %[[H:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Eh"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ex"}
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! SPLIT: %[[EV:.*]] = fir.load %[[E]]#0
! SPLIT: %[[FV:.*]] = fir.load %[[F]]#0
! SPLIT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! SPLIT: %[[GV:.*]] = fir.load %[[G]]#0
! SPLIT: %[[HV:.*]] = fir.load %[[H]]#0
! SPLIT: %[[GH:.*]] = arith.mulf %[[GV]], %[[HV]]
! SPLIT: %[[EFGH:.*]] = arith.addf %[[EF]], %[[GH]]
! SPLIT: %[[TAIL:.*]] = arith.addf %[[CD]], %[[EFGH]]
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AB]]
! SPLIT-NOT: arith.addf %[[HEAD]], %[[CD]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_self_update4
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ea"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Eb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ec"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ed"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ee"}
! DEFAULT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ef"}
! DEFAULT-DAG: %[[G:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Eg"}
! DEFAULT-DAG: %[[H:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Eh"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update4Ex"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! DEFAULT: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! DEFAULT: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! DEFAULT: %[[EV:.*]] = fir.load %[[E]]#0
! DEFAULT: %[[FV:.*]] = fir.load %[[F]]#0
! DEFAULT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! DEFAULT: %[[XABCDEF:.*]] = arith.addf %[[XABCD]], %[[EF]]
! DEFAULT: %[[GV:.*]] = fir.load %[[G]]#0
! DEFAULT: %[[HV:.*]] = fir.load %[[H]]#0
! DEFAULT: %[[GH:.*]] = arith.mulf %[[GV]], %[[HV]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABCDEF]], %[[GH]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! Default:   (((a*b + c*d) + e*f) + g*h)
! Rewritten: ((e*f + g*h) + (a*b + c*d))
subroutine eligible_out_of_place4(y,a,b,c,d,e,f,g,h)
  real(8) :: y,a,b,c,d,e,f,g,h
  y = a*b + c*d + e*f + g*h
end

! SPLIT-LABEL: func.func @_QPeligible_out_of_place4
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ea"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Eb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ec"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ed"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ee"}
! SPLIT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ef"}
! SPLIT-DAG: %[[G:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Eg"}
! SPLIT-DAG: %[[H:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Eh"}
! SPLIT-DAG: %[[Y:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ey"}
! SPLIT: %[[EV:.*]] = fir.load %[[E]]#0
! SPLIT: %[[FV:.*]] = fir.load %[[F]]#0
! SPLIT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! SPLIT: %[[GV:.*]] = fir.load %[[G]]#0
! SPLIT: %[[HV:.*]] = fir.load %[[H]]#0
! SPLIT: %[[GH:.*]] = arith.mulf %[[GV]], %[[HV]]
! SPLIT: %[[TAIL:.*]] = arith.addf %[[EF]], %[[GH]]
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[AB]], %[[CD]]
! SPLIT-NOT: arith.addf %[[HEAD]], %[[EF]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[Y]]#0

! DEFAULT-LABEL: func.func @_QPeligible_out_of_place4
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ea"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Eb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ec"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ed"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ee"}
! DEFAULT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ef"}
! DEFAULT-DAG: %[[G:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Eg"}
! DEFAULT-DAG: %[[H:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Eh"}
! DEFAULT-DAG: %[[Y:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_out_of_place4Ey"}
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! DEFAULT: %[[ABCD:.*]] = arith.addf %[[AB]], %[[CD]]
! DEFAULT: %[[EV:.*]] = fir.load %[[E]]#0
! DEFAULT: %[[FV:.*]] = fir.load %[[F]]#0
! DEFAULT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! DEFAULT: %[[ABCDEF:.*]] = arith.addf %[[ABCD]], %[[EF]]
! DEFAULT: %[[GV:.*]] = fir.load %[[G]]#0
! DEFAULT: %[[HV:.*]] = fir.load %[[H]]#0
! DEFAULT: %[[GH:.*]] = arith.mulf %[[GV]], %[[HV]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[ABCDEF]], %[[GH]]
! DEFAULT: hlfir.assign %[[RES]] to %[[Y]]#0

! Default:   (((x + a) + b*c) + d*e)
! Rewritten: ((b*c + d*e) + (x + a))
subroutine eligible_scalar_term(x,a,b,c,d,e)
  real(8) :: x,a,b,c,d,e
  x = x + a + b*c + d*e
end

! SPLIT-LABEL: func.func @_QPeligible_scalar_term
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEd"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEe"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEx"}
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[BC:.*]] = arith.mulf %[[BV]], %[[CV]]
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[EV:.*]] = fir.load %[[E]]#0
! SPLIT: %[[DE:.*]] = arith.mulf %[[DV]], %[[EV]]
! SPLIT: %[[TAIL:.*]] = arith.addf %[[BC]], %[[DE]]
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AV]]
! SPLIT-NOT: arith.addf %[[HEAD]], %[[BC]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_scalar_term
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEd"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEe"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_scalar_termEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = arith.addf %[[XV]], %[[AV]]
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[BC:.*]] = arith.mulf %[[BV]], %[[CV]]
! DEFAULT: %[[XABC:.*]] = arith.addf %[[XA]], %[[BC]]
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[EV:.*]] = fir.load %[[E]]#0
! DEFAULT: %[[DE:.*]] = arith.mulf %[[DV]], %[[EV]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABC]], %[[DE]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

subroutine guard_parentheses(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = (x + a*b) + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_parentheses
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_parenthesesEx"}
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_parenthesesEe"}
! NO-REWRITE: fir.load %[[X]]#0
! NO-REWRITE: hlfir.no_reassoc
! NO-REWRITE: fir.load %[[E]]#0

subroutine guard_subtract(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x - a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_subtract
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEb"}
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEc"}
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEd"}
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEe"}
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEf"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_subtractEx"}
! NO-REWRITE: %[[XV:.*]] = fir.load %[[X]]#0
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.subf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[DV:.*]] = fir.load %[[D]]#0
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %[[E]]#0
! NO-REWRITE: %[[FV:.*]] = fir.load %[[F]]#0
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

real(8) function foo(a)
  real(8) :: a
  foo = a
end

subroutine guard_call(x,a,b,c,d,e)
  real(8) :: x,a,b,c,d,e,foo
  x = x + foo(a) + b*c + d*e
end

! NO-REWRITE-LABEL: func.func @_QPguard_call
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_callEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_callEb"}
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_callEc"}
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_callEd"}
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_callEe"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_callEx"}
! NO-REWRITE: %[[XV:.*]] = fir.load %[[X]]#0
! NO-REWRITE: %[[FOO:.*]] = fir.call @_QPfoo(%[[A]]#0)
! NO-REWRITE: %[[XFOO:.*]] = arith.addf %[[XV]], %[[FOO]]
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[BC:.*]] = arith.mulf %[[BV]], %[[CV]]
! NO-REWRITE: %[[XFOOBC:.*]] = arith.addf %[[XFOO]], %[[BC]]
! NO-REWRITE: %[[DV:.*]] = fir.load %[[D]]#0
! NO-REWRITE: %[[EV:.*]] = fir.load %[[E]]#0
! NO-REWRITE: %[[DE:.*]] = arith.mulf %[[DV]], %[[EV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XFOOBC]], %[[DE]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

subroutine guard_array(n,x,a,b,c,d,e,f)
  integer :: n
  real(8) :: x(n),a(n),b(n),c(n),d(n),e(n),f(n)
  x = x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_array
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEb"}
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEc"}
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEd"}
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEe"}
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEf"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_arrayEx"}
! NO-REWRITE: %[[AB:.*]] = hlfir.elemental
! NO-REWRITE: fir.load
! NO-REWRITE: fir.load
! NO-REWRITE: %[[ABV:.*]] = arith.mulf
! NO-REWRITE: hlfir.yield_element %[[ABV]]
! NO-REWRITE: %[[XAB:.*]] = hlfir.elemental
! NO-REWRITE: hlfir.designate %[[X]]#0
! NO-REWRITE: %[[ABAPPLY:.*]] = hlfir.apply %[[AB]]
! NO-REWRITE: %[[XV:.*]] = fir.load
! NO-REWRITE: %[[XABV:.*]] = arith.addf %[[XV]], %[[ABAPPLY]]
! NO-REWRITE: hlfir.yield_element %[[XABV]]
! NO-REWRITE: %[[CD:.*]] = hlfir.elemental
! NO-REWRITE: fir.load
! NO-REWRITE: fir.load
! NO-REWRITE: %[[CDV:.*]] = arith.mulf
! NO-REWRITE: hlfir.yield_element %[[CDV]]
! NO-REWRITE: %[[XABCD:.*]] = hlfir.elemental
! NO-REWRITE: %[[XABAPPLY:.*]] = hlfir.apply %[[XAB]]
! NO-REWRITE: %[[CDAPPLY:.*]] = hlfir.apply %[[CD]]
! NO-REWRITE: %[[XABCDV:.*]] = arith.addf %[[XABAPPLY]], %[[CDAPPLY]]
! NO-REWRITE: hlfir.yield_element %[[XABCDV]]
! NO-REWRITE: %[[EF:.*]] = hlfir.elemental
! NO-REWRITE: fir.load
! NO-REWRITE: fir.load
! NO-REWRITE: %[[EFV:.*]] = arith.mulf
! NO-REWRITE: hlfir.yield_element %[[EFV]]
! NO-REWRITE: %[[XABCDEF:.*]] = hlfir.elemental
! NO-REWRITE: %[[XABCDAPPLY:.*]] = hlfir.apply %[[XABCD]]
! NO-REWRITE: %[[EFAPPLY:.*]] = hlfir.apply %[[EF]]
! NO-REWRITE: %[[XABCDEFV:.*]] = arith.addf %[[XABCDAPPLY]], %[[EFAPPLY]]
! NO-REWRITE: hlfir.yield_element %[[XABCDEFV]]
! NO-REWRITE: hlfir.assign %[[XABCDEF]] to %[[X]]#0

subroutine guard_short_sum(x,a,b)
  real(8) :: x,a,b
  x = x + a*b
end

! NO-REWRITE-LABEL: func.func @_QPguard_short_sum
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_short_sumEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_short_sumEb"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_short_sumEx"}
! NO-REWRITE: %[[XV:.*]] = fir.load %[[X]]#0
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

subroutine guard_mixed_kind(x,a,b,c,d,e,f)
  real(8) :: x
  real(4) :: a,b,c,d,e,f
  x = a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_mixed_kind
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEb"}
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEc"}
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEd"}
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEe"}
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEf"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_mixed_kindEx"}
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[DV:.*]] = fir.load %[[D]]#0
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[ABCD:.*]] = arith.addf %[[AB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %[[E]]#0
! NO-REWRITE: %[[FV:.*]] = fir.load %[[F]]#0
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[SUM:.*]] = arith.addf %[[ABCD]], %[[EF]]
! NO-REWRITE: %[[RES:.*]] = fir.convert %[[SUM]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

module split_sum_guard_mod
  real(8), volatile :: use_volatile_x
  real(8), asynchronous :: use_asynchronous_x
end module

subroutine guard_use_assoc_volatile(y,a,b,c,d,e,f)
  use split_sum_guard_mod
  real(8) :: y,a,b,c,d,e,f
  y = use_volatile_x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_use_assoc_volatile
! NO-REWRITE: %[[XV:.*]] = fir.load %{{.*}} : !fir.ref<f64, volatile>
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_use_assoc_asynchronous(y,a,b,c,d,e,f)
  use split_sum_guard_mod
  real(8) :: y,a,b,c,d,e,f
  y = use_asynchronous_x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_use_assoc_asynchronous
! NO-REWRITE: %[[XV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_volatile(x,a,b,c,d,e,f)
  real(8), volatile :: x
  real(8) :: a,b,c,d,e,f
  x = x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_volatile
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEa"
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEb"
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEc"
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEd"
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEe"
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEf"
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatileEx"
! NO-REWRITE: %[[XV:.*]] = fir.load %[[X]]#0
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[DV:.*]] = fir.load %[[D]]#0
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %[[E]]#0
! NO-REWRITE: %[[FV:.*]] = fir.load %[[F]]#0
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

subroutine guard_volatile_lhs_only(x,a,b,c,d,e,f)
  real(8), volatile :: x
  real(8) :: a,b,c,d,e,f
  x = a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_volatile_lhs_only
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEa"
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEb"
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEc"
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEd"
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEe"
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEf"
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_volatile_lhs_onlyEx"
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[DV:.*]] = fir.load %[[D]]#0
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[ABCD:.*]] = arith.addf %[[AB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %[[E]]#0
! NO-REWRITE: %[[FV:.*]] = fir.load %[[F]]#0
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[ABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

subroutine guard_asynchronous(x,a,b,c,d,e,f)
  real(8), asynchronous :: x
  real(8) :: a,b,c,d,e,f
  x = x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_asynchronous
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEa"
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEb"
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEc"
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEd"
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEe"
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEf"
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFguard_asynchronousEx"
! NO-REWRITE: %[[XV:.*]] = fir.load %[[X]]#0
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[DV:.*]] = fir.load %[[D]]#0
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %[[E]]#0
! NO-REWRITE: %[[FV:.*]] = fir.load %[[F]]#0
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

subroutine guard_volatile_array_element(i,x,a,b,c,d,e,f)
  integer :: i
  real(8), volatile :: x(10)
  real(8) :: a,b,c,d,e,f
  x(i) = x(i) + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_volatile_array_element
! NO-REWRITE: %[[XELT:.*]] = hlfir.designate {{.*}} -> !fir.ref<f64, volatile>
! NO-REWRITE: %[[XV:.*]] = fir.load %[[XELT]]
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_volatile_subscript(i,x,a,b,c,d,e,f)
  integer, volatile :: i
  real(8) :: x(10),a,b,c,d,e,f
  x(i) = x(i) + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_volatile_subscript
! NO-REWRITE: %[[IV:.*]] = fir.load %{{.*}} : !fir.ref<i32, volatile>
! NO-REWRITE: %[[SUB:.*]] = fir.convert %[[IV]]
! NO-REWRITE: %[[XELT:.*]] = hlfir.designate {{.*}}(%[[SUB]])
! NO-REWRITE: %[[XV:.*]] = fir.load %[[XELT]]
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_associate_volatile_array_element(i,x,y,a,b,c,d,e,f)
  integer :: i
  real(8), volatile :: x(10)
  real(8) :: y,a,b,c,d,e,f
  associate(v => x(i))
    y = v + a*b + c*d + e*f
  end associate
end

! NO-REWRITE-LABEL: func.func @_QPguard_associate_volatile_array_element
! NO-REWRITE: %[[VELT:.*]] = hlfir.designate {{.*}} -> !fir.ref<f64, volatile>
! NO-REWRITE: %[[VV:.*]] = fir.load %{{.*}} : !fir.ref<f64, volatile>
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[VAB:.*]] = arith.addf %[[VV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[VABCD:.*]] = arith.addf %[[VAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[VABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_associate_asynchronous_array_element(i,x,y,a,b,c,d,e,f)
  integer :: i
  real(8), asynchronous :: x(10)
  real(8) :: y,a,b,c,d,e,f
  associate(v => x(i))
    y = v + a*b + c*d + e*f
  end associate
end

! NO-REWRITE-LABEL: func.func @_QPguard_associate_asynchronous_array_element
! NO-REWRITE: %[[VELT:.*]] = hlfir.designate {{.*}} -> !fir.ref<f64>
! NO-REWRITE: %[[VV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[VAB:.*]] = arith.addf %[[VV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[VABCD:.*]] = arith.addf %[[VAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[VABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_volatile_complex_part(x,z,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  complex(8), volatile :: z
  x = z%re + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_volatile_complex_part
! NO-REWRITE: %[[ZRE_REF:.*]] = hlfir.designate {{.*}} real
! NO-REWRITE: %[[ZRE:.*]] = fir.load %[[ZRE_REF]]
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[ZAB:.*]] = arith.addf %[[ZRE]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[ZABCD:.*]] = arith.addf %[[ZAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[ZABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_associate_volatile_complex_part(x,z,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  complex(8), volatile :: z
  associate(v => z%re)
    x = v + a*b + c*d + e*f
  end associate
end

! NO-REWRITE-LABEL: func.func @_QPguard_associate_volatile_complex_part
! NO-REWRITE: %[[ZRE_REF:.*]] = hlfir.designate {{.*}} real
! NO-REWRITE: %[[ZRE:.*]] = fir.load %{{.*}} : !fir.ref<f64, volatile>
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[ZAB:.*]] = arith.addf %[[ZRE]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[ZABCD:.*]] = arith.addf %[[ZAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[ZABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_associate_asynchronous_complex_part(x,z,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  complex(8), asynchronous :: z
  associate(v => z%re)
    x = v + a*b + c*d + e*f
  end associate
end

! NO-REWRITE-LABEL: func.func @_QPguard_associate_asynchronous_complex_part
! NO-REWRITE: %[[ZRE_REF:.*]] = hlfir.designate {{.*}} real
! NO-REWRITE: %[[ZRE:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[ZAB:.*]] = arith.addf %[[ZRE]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[ZABCD:.*]] = arith.addf %[[ZAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[ZABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_asynchronous_complex_part(x,z,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  complex(8), asynchronous :: z
  x = z%re + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_asynchronous_complex_part
! NO-REWRITE: %[[ZRE_REF:.*]] = hlfir.designate {{.*}} real
! NO-REWRITE: %[[ZRE:.*]] = fir.load %[[ZRE_REF]]
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[ZAB:.*]] = arith.addf %[[ZRE]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[ZABCD:.*]] = arith.addf %[[ZAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[ZABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}

subroutine guard_asynchronous_array_element(i,x,a,b,c,d,e,f)
  integer :: i
  real(8), asynchronous :: x(10)
  real(8) :: a,b,c,d,e,f
  x(i) = x(i) + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_asynchronous_array_element
! NO-REWRITE: %[[XELT:.*]] = hlfir.designate {{.*}} -> !fir.ref<f64>
! NO-REWRITE: %[[XV:.*]] = fir.load %[[XELT]]
! NO-REWRITE: %[[AV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[BV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[DV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[FV:.*]] = fir.load %{{.*}} : !fir.ref<f64>
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: hlfir.assign %[[RES]] to %{{.*}}
