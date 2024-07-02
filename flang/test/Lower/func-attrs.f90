! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

pure subroutine sub1()
end

! CHECK: func.func @_QPsub1() attributes {fir.func_pure}

elemental subroutine sub2()
end

! CHECK: func.func @_QPsub2() attributes {fir.func_elemental, fir.func_pure}

recursive subroutine sub3()
end

! CHECK: func.func @_QPsub3() attributes {fir.func_recursive}

pure function fct1()
end

! CHECK: func.func @_QPfct1() -> f32 attributes {fir.func_pure}

elemental function fct2()
end

! CHECK: func.func @_QPfct2() -> f32 attributes {fir.func_elemental, fir.func_pure}

recursive function fct3()
end

! CHECK: func.func @_QPfct3() -> f32 attributes {fir.func_recursive}
