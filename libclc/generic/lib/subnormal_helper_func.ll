@__CLC_SUBNORMAL_DISABLE = external global i1

define i1 @__clc_subnormals_disabled() #0 {
  %disable = load i1, i1* @__CLC_SUBNORMAL_DISABLE
  ret i1 %disable
}

attributes #0 = { alwaysinline }
