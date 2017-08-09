//hadd = (x+y)>>1
//This can be simplified to x>>1 + y>>1 + (1 if both x and y have the 1s bit set)
//This saves us having to do any checks for overflow in the addition sum
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE hadd(__CLC_GENTYPE x, __CLC_GENTYPE y) {
    return (x>>(__CLC_GENTYPE)1)+(y>>(__CLC_GENTYPE)1)+(x&y&(__CLC_GENTYPE)1);
}
