! RUN: %flang_fc1 -emit-hlfir -ffp-sum-reassociation -o - %s | FileCheck %s --check-prefixes=SPLIT,NO-REWRITE --implicit-check-not=arith.negf
! RUN: %flang_fc1 -emit-hlfir -fno-fp-sum-reassociation -o - %s | FileCheck %s --check-prefixes=DEFAULT,NO-REWRITE
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

! Default:   (((x + (a-b)) + (c-d)) + (e-f))
! Rewritten: ((c-d) + (e-f)) + (x + (a-b))
subroutine eligible_parenthesized_subtractions(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x + (a-b) + (c-d) + (e-f)
end

! SPLIT-LABEL: func.func @_QPeligible_parenthesized_subtractions
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEd"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEe"}
! SPLIT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEf"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEx"}
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[CD_SUB:.*]] = arith.subf %[[CV]], %[[DV]]
! SPLIT: %[[CD:.*]] = hlfir.no_reassoc %[[CD_SUB]]
! SPLIT: %[[EV:.*]] = fir.load %[[E]]#0
! SPLIT: %[[FV:.*]] = fir.load %[[F]]#0
! SPLIT: %[[EF_SUB:.*]] = arith.subf %[[EV]], %[[FV]]
! SPLIT: %[[EF:.*]] = hlfir.no_reassoc %[[EF_SUB]]
! SPLIT: %[[TAIL:.*]] = arith.addf %[[CD]], %[[EF]]
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[AB_SUB:.*]] = arith.subf %[[AV]], %[[BV]]
! SPLIT: %[[AB:.*]] = hlfir.no_reassoc %[[AB_SUB]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AB]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_parenthesized_subtractions
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEd"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEe"}
! DEFAULT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEf"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_subtractionsEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[AB_SUB:.*]] = arith.subf %[[AV]], %[[BV]]
! DEFAULT: %[[AB:.*]] = hlfir.no_reassoc %[[AB_SUB]]
! DEFAULT: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[CD_SUB:.*]] = arith.subf %[[CV]], %[[DV]]
! DEFAULT: %[[CD:.*]] = hlfir.no_reassoc %[[CD_SUB]]
! DEFAULT: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! DEFAULT: %[[EV:.*]] = fir.load %[[E]]#0
! DEFAULT: %[[FV:.*]] = fir.load %[[F]]#0
! DEFAULT: %[[EF_SUB:.*]] = arith.subf %[[EV]], %[[FV]]
! DEFAULT: %[[EF:.*]] = hlfir.no_reassoc %[[EF_SUB]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! The parenthesized addition is moved as one opaque term; its inner Add is not
! part of the top-level additive spine.
! Default:   (((x + (a+b)) + c*d) + e*f)
! Rewritten: ((c*d + e*f) + (x + (a+b)))
subroutine eligible_parenthesized_add(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x + (a+b) + c*d + e*f
end

! SPLIT-LABEL: func.func @_QPeligible_parenthesized_add
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEd"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEe"}
! SPLIT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEf"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEx"}
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
! SPLIT: %[[AB_ADD:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[AB:.*]] = hlfir.no_reassoc %[[AB_ADD]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AB]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_parenthesized_add
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEd"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEe"}
! DEFAULT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEf"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_parenthesized_addEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[AB_ADD:.*]] = arith.addf %[[AV]], %[[BV]]
! DEFAULT: %[[AB:.*]] = hlfir.no_reassoc %[[AB_ADD]]
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

! A Parentheses root is not a top-level Add and therefore is not rewritten.
! Default:   ((((x + a*b) + c*d) + e*f))
! Rewritten: ((((x + a*b) + c*d) + e*f))
subroutine guard_whole_rhs_parentheses(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = (x + a*b + c*d + e*f)
end

! NO-REWRITE-LABEL: func.func @_QPguard_whole_rhs_parentheses
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEb"}
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEc"}
! NO-REWRITE-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEd"}
! NO-REWRITE-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEe"}
! NO-REWRITE-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEf"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_whole_rhs_parenthesesEx"}
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
! NO-REWRITE: %[[SUM:.*]] = arith.addf %[[XABCD]], %[[EF]]
! NO-REWRITE: %[[PAREN:.*]] = hlfir.no_reassoc %[[SUM]]
! NO-REWRITE: hlfir.assign %[[PAREN]] to %[[X]]#0

! The unparenthesized Subtract is flattened into separate positive and negative
! terms instead of remaining an opaque head term.
! Default:   (((x - a*b) + c*d) + e*f)
! Rewritten: (c*d + e*f) + (x - a*b)
subroutine eligible_signed_subtract(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x - a*b + c*d + e*f
end

! SPLIT-LABEL: func.func @_QPeligible_signed_subtract
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEd"}
! SPLIT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEe"}
! SPLIT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEf"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEx"}
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
! SPLIT: %[[HEAD:.*]] = arith.subf %[[XV]], %[[AB]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_signed_subtract
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEd"}
! DEFAULT-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEe"}
! DEFAULT-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEf"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_signed_subtractEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! DEFAULT: %[[XAB:.*]] = arith.subf %[[XV]], %[[AB]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! DEFAULT: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]]
! DEFAULT: %[[EV:.*]] = fir.load %[[E]]#0
! DEFAULT: %[[FV:.*]] = fir.load %[[F]]#0
! DEFAULT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABCD]], %[[EF]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! The tail starts negative. Rebuild -b+c as -(b-c), then use the explicitly
! permitted -X+Y -> Y-X reassociation to avoid a unary negation:
! Default:   (((x + a) - b) + c)
! Rewritten: (x + a) - (b - c)
subroutine eligible_leading_negative_tail(x,a,b,c)
  real(8) :: x,a,b,c
  x = x + a - b + c
end

! SPLIT-LABEL: func.func @_QPeligible_leading_negative_tail
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEc"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEx"}
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AV]]
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[TAIL:.*]] = arith.subf %[[BV]], %[[CV]]
! SPLIT: %[[RES:.*]] = arith.subf %[[HEAD]], %[[TAIL]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_leading_negative_tail
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEc"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_leading_negative_tailEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = arith.addf %[[XV]], %[[AV]]
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[XAB:.*]] = arith.subf %[[XA]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[RES:.*]] = arith.addf %[[XAB]], %[[CV]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! The tail ends negative and can be rebuilt directly with Subtract.
! Default:   (((x + a) + b) - c)
! Rewritten: (b - c) + (x + a)
subroutine eligible_trailing_negative_tail(x,a,b,c)
  real(8) :: x,a,b,c
  x = x + a + b - c
end

! SPLIT-LABEL: func.func @_QPeligible_trailing_negative_tail
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEc"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEx"}
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[TAIL:.*]] = arith.subf %[[BV]], %[[CV]]
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AV]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_trailing_negative_tail
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEc"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_trailing_negative_tailEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = arith.addf %[[XV]], %[[AV]]
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[XAB:.*]] = arith.addf %[[XA]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[RES:.*]] = arith.subf %[[XAB]], %[[CV]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! A root Subtract and consecutive negative terms are eligible. The entirely
! negative tail is represented by its positive magnitude, without unary minus.
! Default:   (((x - a) - b) - c)
! Rewritten: (x - a) - (b + c)
subroutine eligible_consecutive_subtraction(x,a,b,c)
  real(8) :: x,a,b,c
  x = x - a - b - c
end

! SPLIT-LABEL: func.func @_QPeligible_consecutive_subtraction
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEc"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEx"}
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = arith.subf %[[XV]], %[[AV]]
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[TAIL:.*]] = arith.addf %[[BV]], %[[CV]]
! SPLIT: %[[RES:.*]] = arith.subf %[[HEAD]], %[[TAIL]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_consecutive_subtraction
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEc"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_consecutive_subtractionEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = arith.subf %[[XV]], %[[AV]]
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[XAB:.*]] = arith.subf %[[XA]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[RES:.*]] = arith.subf %[[XAB]], %[[CV]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! Nested unparenthesized Add and Subtract nodes all contribute signed terms.
! Default:   ((((x - a) + b) - c) + d)
! Rewritten: (b - (c - d)) + (x - a)
subroutine eligible_nested_unparenthesized_subtraction(x,a,b,c,d)
  real(8) :: x,a,b,c,d
  x = x - a + b - c + d
end

! SPLIT-LABEL: func.func @_QPeligible_nested_unparenthesized_subtraction
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEd"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEx"}
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[CD:.*]] = arith.subf %[[CV]], %[[DV]]
! SPLIT: %[[TAIL:.*]] = arith.subf %[[BV]], %[[CD]]
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = arith.subf %[[XV]], %[[AV]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_nested_unparenthesized_subtraction
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEd"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_nested_unparenthesized_subtractionEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = arith.subf %[[XV]], %[[AV]]
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[XAB:.*]] = arith.addf %[[XA]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[XABC:.*]] = arith.subf %[[XAB]], %[[CV]]
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABC]], %[[DV]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! Complex addition and subtraction use the same signed-term split. The
! parenthesized c-d remains one opaque no_reassoc value.
! Default:   (((x - a) + b) - (c-d))
! Rewritten: (b - (c-d)) + (x - a)
subroutine eligible_complex_signed_parenthesized(x,a,b,c,d)
  complex(4) :: x,a,b,c,d
  x = x - a + b - (c-d)
end

! SPLIT-LABEL: func.func @_QPeligible_complex_signed_parenthesized
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEd"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEx"}
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[CD_SUB:.*]] = fir.subc %[[CV]], %[[DV]] {{.*}} : complex<f32>
! SPLIT: %[[CD:.*]] = hlfir.no_reassoc %[[CD_SUB]] : complex<f32>
! SPLIT: %[[TAIL:.*]] = fir.subc %[[BV]], %[[CD]] {{.*}} : complex<f32>
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = fir.subc %[[XV]], %[[AV]] {{.*}} : complex<f32>
! SPLIT: %[[RES:.*]] = fir.addc %[[TAIL]], %[[HEAD]] {{.*}} : complex<f32>
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_complex_signed_parenthesized
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEd"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_signed_parenthesizedEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = fir.subc %[[XV]], %[[AV]] {{.*}} : complex<f32>
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[XAB:.*]] = fir.addc %[[XA]], %[[BV]] {{.*}} : complex<f32>
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[CD_SUB:.*]] = fir.subc %[[CV]], %[[DV]] {{.*}} : complex<f32>
! DEFAULT: %[[CD:.*]] = hlfir.no_reassoc %[[CD_SUB]] : complex<f32>
! DEFAULT: %[[RES:.*]] = fir.subc %[[XAB]], %[[CD]] {{.*}} : complex<f32>
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

! A second complex kind exercises category dispatch independently of kind.
! Default:   (((x + a) + b) + c)
! Rewritten: (b + c) + (x + a)
subroutine eligible_complex_kind8(x,a,b,c)
  complex(8) :: x,a,b,c
  x = x + a + b + c
end

! SPLIT-LABEL: func.func @_QPeligible_complex_kind8
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_kind8Ea"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_kind8Eb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_kind8Ec"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_complex_kind8Ex"}
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[TAIL:.*]] = fir.addc %[[BV]], %[[CV]] {{.*}} : complex<f64>
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = fir.addc %[[XV]], %[[AV]] {{.*}} : complex<f64>
! SPLIT: %[[RES:.*]] = fir.addc %[[TAIL]], %[[HEAD]] {{.*}} : complex<f64>
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! It isn't as useful to re-write integer expressions because the middle-end can
! already re-associate them somewhat (within the bounds of avoiding overflow).
subroutine guard_integer(x,a,b,c)
  integer :: x,a,b,c
  x = x + a - b + c
end

! NO-REWRITE-LABEL: func.func @_QPguard_integer
! NO-REWRITE-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_integerEa"}
! NO-REWRITE-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_integerEb"}
! NO-REWRITE-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_integerEc"}
! NO-REWRITE-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFguard_integerEx"}
! NO-REWRITE: %[[XV:.*]] = fir.load %[[X]]#0
! NO-REWRITE: %[[AV:.*]] = fir.load %[[A]]#0
! NO-REWRITE: %[[XA:.*]] = arith.addi %[[XV]], %[[AV]]
! NO-REWRITE: %[[BV:.*]] = fir.load %[[B]]#0
! NO-REWRITE: %[[XAB:.*]] = arith.subi %[[XA]], %[[BV]]
! NO-REWRITE: %[[CV:.*]] = fir.load %[[C]]#0
! NO-REWRITE: %[[RES:.*]] = arith.addi %[[XAB]], %[[CV]]
! NO-REWRITE: hlfir.assign %[[RES]] to %[[X]]#0

! Subtraction immediately outside a parenthesized term changes the term's
! outer sign, but the parenthesized b-c remains one opaque no_reassoc value.
! Default:   (((x + a) - (b-c)) + d)
! Rewritten: (x + a) - ((b-c) - d)
subroutine eligible_subtract_parenthesized_term(x,a,b,c,d)
  real(8) :: x,a,b,c,d
  x = x + a - (b-c) + d
end

! SPLIT-LABEL: func.func @_QPeligible_subtract_parenthesized_term
! SPLIT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEa"}
! SPLIT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEb"}
! SPLIT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEc"}
! SPLIT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEd"}
! SPLIT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEx"}
! SPLIT: %[[XV:.*]] = fir.load %[[X]]#0
! SPLIT: %[[AV:.*]] = fir.load %[[A]]#0
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[AV]]
! SPLIT: %[[BV:.*]] = fir.load %[[B]]#0
! SPLIT: %[[CV:.*]] = fir.load %[[C]]#0
! SPLIT: %[[BC_SUB:.*]] = arith.subf %[[BV]], %[[CV]]
! SPLIT: %[[BC:.*]] = hlfir.no_reassoc %[[BC_SUB]]
! SPLIT: %[[DV:.*]] = fir.load %[[D]]#0
! SPLIT: %[[TAIL:.*]] = arith.subf %[[BC]], %[[DV]]
! SPLIT: %[[RES:.*]] = arith.subf %[[HEAD]], %[[TAIL]]
! SPLIT: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_subtract_parenthesized_term
! DEFAULT-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEa"}
! DEFAULT-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEb"}
! DEFAULT-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEc"}
! DEFAULT-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEd"}
! DEFAULT-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_subtract_parenthesized_termEx"}
! DEFAULT: %[[XV:.*]] = fir.load %[[X]]#0
! DEFAULT: %[[AV:.*]] = fir.load %[[A]]#0
! DEFAULT: %[[XA:.*]] = arith.addf %[[XV]], %[[AV]]
! DEFAULT: %[[BV:.*]] = fir.load %[[B]]#0
! DEFAULT: %[[CV:.*]] = fir.load %[[C]]#0
! DEFAULT: %[[BC_SUB:.*]] = arith.subf %[[BV]], %[[CV]]
! DEFAULT: %[[BC:.*]] = hlfir.no_reassoc %[[BC_SUB]]
! DEFAULT: %[[XABC:.*]] = arith.subf %[[XA]], %[[BC]]
! DEFAULT: %[[DV:.*]] = fir.load %[[D]]#0
! DEFAULT: %[[RES:.*]] = arith.addf %[[XABC]], %[[DV]]
! DEFAULT: hlfir.assign %[[RES]] to %[[X]]#0

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

! Default:   (((x + sqrt((a+b)+c)) + d*e) + f*g)
! Rewritten: ((d*e + f*g) + (x + sqrt((a+b)+c)))
! A qualifying outer sum is rewritten once. The pure call remains an opaque
! term, so its additive argument retains source order.
subroutine eligible_pure_call(x,a,b,c,d,e,f,g)
  real(8) :: x,a,b,c,d,e,f,g
  x = x + sqrt(a+b+c) + d*e + f*g
end

! SPLIT-LABEL: func.func @_QPeligible_pure_call
! SPLIT: %[[DV:.*]] = fir.load
! SPLIT: %[[EV:.*]] = fir.load
! SPLIT: %[[DE:.*]] = arith.mulf %[[DV]], %[[EV]]
! SPLIT: %[[FV:.*]] = fir.load
! SPLIT: %[[GV:.*]] = fir.load
! SPLIT: %[[FG:.*]] = arith.mulf %[[FV]], %[[GV]]
! SPLIT: %[[TAIL:.*]] = arith.addf %[[DE]], %[[FG]]
! SPLIT: %[[XV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[ABC:.*]] = arith.addf %[[AB]], %[[CV]]
! SPLIT: %[[CALL:.*]] = math.sqrt %[[ABC]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[CALL]]
! SPLIT: %[[RES:.*]] = arith.addf %[[TAIL]], %[[HEAD]]
! SPLIT: hlfir.assign %[[RES]]

! DEFAULT-LABEL: func.func @_QPeligible_pure_call
! DEFAULT: %[[XV:.*]] = fir.load
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[ABC:.*]] = arith.addf %[[AB]], %[[CV]]
! DEFAULT: %[[CALL:.*]] = math.sqrt %[[ABC]]
! DEFAULT: %[[HEAD:.*]] = arith.addf %[[XV]], %[[CALL]]
! DEFAULT: %[[DV:.*]] = fir.load
! DEFAULT: %[[EV:.*]] = fir.load
! DEFAULT: %[[DE:.*]] = arith.mulf %[[DV]], %[[EV]]
! DEFAULT: %[[HEAD_DE:.*]] = arith.addf %[[HEAD]], %[[DE]]
! DEFAULT: %[[FV:.*]] = fir.load
! DEFAULT: %[[GV:.*]] = fir.load
! DEFAULT: %[[FG:.*]] = arith.mulf %[[FV]], %[[GV]]
! DEFAULT: %[[RES:.*]] = arith.addf %[[HEAD_DE]], %[[FG]]
! DEFAULT: hlfir.assign %[[RES]]

subroutine guard_pure_call_volatile_arg(x,v,a,b,c,d)
  real(8) :: x,a,b,c,d
  real(8), volatile :: v
  x = x + sqrt(v) + a*b + c*d
end

! NO-REWRITE-LABEL: func.func @_QPguard_pure_call_volatile_arg
! NO-REWRITE: %[[XV:.*]] = fir.load
! NO-REWRITE: %[[CALL:.*]] = math.sqrt
! NO-REWRITE: %[[HEAD:.*]] = arith.addf %[[XV]], %[[CALL]]
! NO-REWRITE: %[[AV:.*]] = fir.load
! NO-REWRITE: %[[BV:.*]] = fir.load
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[HEAD_AB:.*]] = arith.addf %[[HEAD]], %[[AB]]
! NO-REWRITE: %[[CV:.*]] = fir.load
! NO-REWRITE: %[[DV:.*]] = fir.load
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[HEAD_AB]], %[[CD]]
! NO-REWRITE: hlfir.assign %[[RES]]

subroutine guard_nested_impure_call(x,a,b,c,d,e)
  real(8) :: x,a,b,c,d,e,foo
  x = x + sqrt(foo(a)) + b*c + d*e
end

! NO-REWRITE-LABEL: func.func @_QPguard_nested_impure_call
! NO-REWRITE: %[[XV:.*]] = fir.load
! NO-REWRITE: %[[IMPURE:.*]] = fir.call @_QPfoo
! NO-REWRITE: %[[PURE:.*]] = math.sqrt %[[IMPURE]]
! NO-REWRITE: %[[HEAD:.*]] = arith.addf %[[XV]], %[[PURE]]
! NO-REWRITE: %[[BV:.*]] = fir.load
! NO-REWRITE: %[[CV:.*]] = fir.load
! NO-REWRITE: %[[BC:.*]] = arith.mulf %[[BV]], %[[CV]]
! NO-REWRITE: %[[HEAD_BC:.*]] = arith.addf %[[HEAD]], %[[BC]]
! NO-REWRITE: %[[DV:.*]] = fir.load
! NO-REWRITE: %[[EV:.*]] = fir.load
! NO-REWRITE: %[[DE:.*]] = arith.mulf %[[DV]], %[[EV]]
! NO-REWRITE: %[[RES:.*]] = arith.addf %[[HEAD_BC]], %[[DE]]
! NO-REWRITE: hlfir.assign %[[RES]]

! Default:   d * real((a+b)+c,8)
! Rewritten: d * real(c+(a+b),8)
subroutine nested_conversion_operand(x,a,b,c,d)
  real(8) :: x,d
  real(4) :: a,b,c
  x = d * real(a+b+c,8)
end

! SPLIT-LABEL: func.func @_QPnested_conversion_operand
! SPLIT: %[[DV:.*]] = fir.load
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[CV]], %[[AB]]
! SPLIT: %[[CONVERT:.*]] = fir.convert %[[SUM]]
! SPLIT: %[[RES:.*]] = arith.mulf %[[DV]], %[[CONVERT]]
! SPLIT: hlfir.assign %[[RES]]

! DEFAULT-LABEL: func.func @_QPnested_conversion_operand
! DEFAULT: %[[DV:.*]] = fir.load
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[SUM:.*]] = arith.addf %[[AB]], %[[CV]]
! DEFAULT: %[[CONVERT:.*]] = fir.convert %[[SUM]]
! DEFAULT: %[[RES:.*]] = arith.mulf %[[DV]], %[[CONVERT]]
! DEFAULT: hlfir.assign %[[RES]]

! Default:   sqrt((a+b)+c)
! Rewritten: sqrt(c+(a+b))
subroutine nested_pure_call_argument(x,a,b,c)
  real(8) :: x,a,b,c
  x = sqrt(a+b+c)
end

! SPLIT-LABEL: func.func @_QPnested_pure_call_argument
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[CV]], %[[AB]]
! SPLIT: %[[SQRT:.*]] = math.sqrt %[[SUM]]
! SPLIT: hlfir.assign %[[SQRT]]

! DEFAULT-LABEL: func.func @_QPnested_pure_call_argument
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[SUM:.*]] = arith.addf %[[AB]], %[[CV]]
! DEFAULT: %[[SQRT:.*]] = math.sqrt %[[SUM]]
! DEFAULT: hlfir.assign %[[SQRT]]

! Default:   atan2((a+b)+c,(d+e)+f)
! Rewritten: atan2(c+(a+b),f+(d+e))
subroutine nested_separate_call_arguments(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = atan2(a+b+c,d+e+f)
end

! SPLIT-LABEL: func.func @_QPnested_separate_call_arguments
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[FIRST:.*]] = arith.addf %[[CV]], %[[AB]]
! SPLIT: %[[FV:.*]] = fir.load
! SPLIT: %[[DV:.*]] = fir.load
! SPLIT: %[[EV:.*]] = fir.load
! SPLIT: %[[DE:.*]] = arith.addf %[[DV]], %[[EV]]
! SPLIT: %[[SECOND:.*]] = arith.addf %[[FV]], %[[DE]]
! SPLIT: math.atan2 %[[FIRST]], %[[SECOND]]

! Default:   (flag ? (a+b)+c : d)
! Rewritten: (flag ? c+(a+b) : d)
subroutine nested_conditional_branch(x,flag,a,b,c,d)
  real(8) :: x,a,b,c,d
  logical :: flag
  x = (flag ? a+b+c : d)
end

! SPLIT-LABEL: func.func @_QPnested_conditional_branch
! SPLIT: fir.if
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[CV]], %[[AB]]
! SPLIT: fir.result %[[SUM]]

! DEFAULT-LABEL: func.func @_QPnested_conditional_branch
! DEFAULT: fir.if
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[SUM:.*]] = arith.addf %[[AB]], %[[CV]]
! DEFAULT: fir.result %[[SUM]]

! Default:   ((a+b)+c > d ? e : f)
! Rewritten: (c+(a+b) > d ? e : f)
subroutine nested_relational_operand(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = (a+b+c > d ? e : f)
end

! SPLIT-LABEL: func.func @_QPnested_relational_operand
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[CV]], %[[AB]]
! SPLIT: arith.cmpf ogt, %[[SUM]]

! DEFAULT-LABEL: func.func @_QPnested_relational_operand
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[SUM:.*]] = arith.addf %[[AB]], %[[CV]]
! DEFAULT: arith.cmpf ogt, %[[SUM]]

subroutine guard_parenthesized_call_argument(x,a,b,c)
  real(8) :: x,a,b,c
  x = sqrt((a+b+c))
end

! NO-REWRITE-LABEL: func.func @_QPguard_parenthesized_call_argument
! NO-REWRITE: %[[AV:.*]] = fir.load
! NO-REWRITE: %[[BV:.*]] = fir.load
! NO-REWRITE: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! NO-REWRITE: %[[CV:.*]] = fir.load
! NO-REWRITE: %[[SUM:.*]] = arith.addf %[[AB]], %[[CV]]
! NO-REWRITE: %[[PAREN:.*]] = hlfir.no_reassoc %[[SUM]]
! NO-REWRITE: %[[SQRT:.*]] = math.sqrt %[[PAREN]]
! NO-REWRITE: hlfir.assign %[[SQRT]]

subroutine guard_short_call_argument(x,a,b)
  real(8) :: x,a,b
  x = sqrt(a+b)
end

! NO-REWRITE-LABEL: func.func @_QPguard_short_call_argument
! NO-REWRITE: %[[AV:.*]] = fir.load
! NO-REWRITE: %[[BV:.*]] = fir.load
! NO-REWRITE: %[[SUM:.*]] = arith.addf %[[AV]], %[[BV]]
! NO-REWRITE: %[[SQRT:.*]] = math.sqrt %[[SUM]]
! NO-REWRITE: hlfir.assign %[[SQRT]]

subroutine guard_non_assignment_context(a,b,c)
  real(8) :: a,b,c
  call consume(a+b+c)
end

! NO-REWRITE-LABEL: func.func @_QPguard_non_assignment_context
! NO-REWRITE: %[[AV:.*]] = fir.load
! NO-REWRITE: %[[BV:.*]] = fir.load
! NO-REWRITE: %[[AB:.*]] = arith.addf %[[AV]], %[[BV]]
! NO-REWRITE: %[[CV:.*]] = fir.load
! NO-REWRITE: %[[SUM:.*]] = arith.addf %[[AB]], %[[CV]]
! NO-REWRITE: fir.call @_QPconsume

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

! The kind conversion remains around the reassociated expression.
! Default:   ((a*b + c*d) + e*f)
! Rewritten: (e*f + (a*b + c*d))
subroutine eligible_whole_real_kind_conversion(x,a,b,c,d,e,f)
  real(8) :: x
  real(4) :: a,b,c,d,e,f
  x = a*b + c*d + e*f
end

! SPLIT-LABEL: func.func @_QPeligible_whole_real_kind_conversion
! SPLIT: %[[EV:.*]] = fir.load
! SPLIT: %[[FV:.*]] = fir.load
! SPLIT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[DV:.*]] = fir.load
! SPLIT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! SPLIT: %[[HEAD:.*]] = arith.addf %[[AB]], %[[CD]]
! SPLIT: %[[SUM:.*]] = arith.addf %[[EF]], %[[HEAD]]
! SPLIT: %[[RES:.*]] = fir.convert %[[SUM]] : (f32) -> f64
! SPLIT: hlfir.assign %[[RES]]

! DEFAULT-LABEL: func.func @_QPeligible_whole_real_kind_conversion
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[DV:.*]] = fir.load
! DEFAULT: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! DEFAULT: %[[HEAD:.*]] = arith.addf %[[AB]], %[[CD]]
! DEFAULT: %[[EV:.*]] = fir.load
! DEFAULT: %[[FV:.*]] = fir.load
! DEFAULT: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]]
! DEFAULT: %[[SUM:.*]] = arith.addf %[[HEAD]], %[[EF]]
! DEFAULT: %[[RES:.*]] = fir.convert %[[SUM]] : (f32) -> f64
! DEFAULT: hlfir.assign %[[RES]]

! Default:   ((a + b) + c)
! Rewritten: (c + (a + b))
subroutine eligible_whole_complex_kind_conversion(x,a,b,c)
  complex(8) :: x
  complex(4) :: a,b,c
  x = a + b + c
end

! SPLIT-LABEL: func.func @_QPeligible_whole_complex_kind_conversion
! SPLIT: %[[CV:.*]] = fir.load
! SPLIT: %[[AV:.*]] = fir.load
! SPLIT: %[[BV:.*]] = fir.load
! SPLIT: %[[HEAD:.*]] = fir.addc %[[AV]], %[[BV]]
! SPLIT: %[[SUM:.*]] = fir.addc %[[CV]], %[[HEAD]]
! SPLIT: %[[RES:.*]] = fir.convert %[[SUM]] : (complex<f32>) -> complex<f64>
! SPLIT: hlfir.assign %[[RES]]

! DEFAULT-LABEL: func.func @_QPeligible_whole_complex_kind_conversion
! DEFAULT: %[[AV:.*]] = fir.load
! DEFAULT: %[[BV:.*]] = fir.load
! DEFAULT: %[[HEAD:.*]] = fir.addc %[[AV]], %[[BV]]
! DEFAULT: %[[CV:.*]] = fir.load
! DEFAULT: %[[SUM:.*]] = fir.addc %[[HEAD]], %[[CV]]
! DEFAULT: %[[RES:.*]] = fir.convert %[[SUM]] : (complex<f32>) -> complex<f64>
! DEFAULT: hlfir.assign %[[RES]]

! A conversion embedded in the additive tree is not yet eligible.
subroutine guard_embedded_kind_conversion(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d
  real(4) :: e,f
  x = a*b + c*d + real(e*f,8)
end

! NO-REWRITE-LABEL: func.func @_QPguard_embedded_kind_conversion
! NO-REWRITE: %[[AV:.*]] = fir.load
! NO-REWRITE: %[[BV:.*]] = fir.load
! NO-REWRITE: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]]
! NO-REWRITE: %[[CV:.*]] = fir.load
! NO-REWRITE: %[[DV:.*]] = fir.load
! NO-REWRITE: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]]
! NO-REWRITE: %[[HEAD:.*]] = arith.addf %[[AB]], %[[CD]]
! NO-REWRITE: %[[EV:.*]] = fir.load
! NO-REWRITE: %[[FV:.*]] = fir.load
! NO-REWRITE: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]] {{.*}} : f32
! NO-REWRITE: %[[CONVERT:.*]] = fir.convert %[[EF]] : (f32) -> f64
! NO-REWRITE: %[[SUM:.*]] = arith.addf %[[HEAD]], %[[CONVERT]]
! NO-REWRITE: hlfir.assign %[[SUM]]

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
