! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

pure subroutine sub1()
end

! CHECK: func.func @_QPsub1() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}

elemental subroutine sub2()
end

! CHECK: func.func @_QPsub2() attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}

non_recursive subroutine sub3()
end

! CHECK: func.func @_QPsub3() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>}

impure elemental subroutine sub4()
end

! CHECK: func.func @_QPsub4() attributes {fir.proc_attrs = #fir.proc_attrs<elemental>}

pure function fct1()
end

! CHECK: func.func @_QPfct1() -> f32 attributes {fir.proc_attrs = #fir.proc_attrs<pure>}

elemental function fct2()
end

! CHECK: func.func @_QPfct2() -> f32 attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>}

non_recursive function fct3()
end

! CHECK: func.func @_QPfct3() -> f32 attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>}
