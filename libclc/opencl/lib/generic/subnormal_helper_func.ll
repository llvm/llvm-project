;;===----------------------------------------------------------------------===;;
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
;;===----------------------------------------------------------------------===;;

@__CLC_SUBNORMAL_DISABLE = external global i1

define i1 @__clc_subnormals_disabled() #0 {
  %disable = load i1, i1* @__CLC_SUBNORMAL_DISABLE
  ret i1 %disable
}

attributes #0 = { alwaysinline }
