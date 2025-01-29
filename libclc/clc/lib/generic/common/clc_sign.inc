// TYPE sign(TYPE x) {
//   if (isnan(x)) {
//     return 0.0F;
//   }
//   if (x > 0.0F) {
//     return 1.0F;
//   }
//   if (x < 0.0F) {
//     return -1.0F;
//   }
//   return x; /* -0.0 or +0.0 */
// }
_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_sign(__CLC_GENTYPE x) {
  __CLC_BIT_INTN x_isnan = __clc_isnan(x);
  __CLC_BIT_INTN x_isgreater_zero = x > __CLC_FP_LIT(0.0);
  __CLC_BIT_INTN x_isless_zero = x < __CLC_FP_LIT(0.0);
  __CLC_GENTYPE sel0 = __clc_select(x, __CLC_FP_LIT(1.0), x_isgreater_zero);
  __CLC_GENTYPE sel1 = __clc_select(sel0, __CLC_FP_LIT(-1.0), x_isless_zero);
  __CLC_GENTYPE sel2 = __clc_select(sel1, __CLC_FP_LIT(0.0), x_isnan);
  return sel2;
}
