#pragma once

extern _Float16 cosf16(_Float16);
extern _Float16 add_and_double(_Float16, _Float16);
extern _Float16 sinf16(_Float16);
extern _Float16 add_and_half(_Float16, _Float16);
extern _Float16 add_and_half_ptr(_Float16, _Float16*);
extern _Float16 add_and_half_ptr_has_no_mask_version(_Float16, _Float16*);
extern _Float16 non_pure_ew_separate_mask(_Float16);
extern float cosf(float);
extern _Float16 mysinf16(_Float16);
extern void side_effect_no_return(float, float);
