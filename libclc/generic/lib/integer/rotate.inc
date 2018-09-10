/**
 * Not necessarily optimal... but it produces correct results (at least for int)
 * If we're lucky, LLVM will recognize the pattern and produce rotate
 * instructions:
 * http://llvm.1065342.n5.nabble.com/rotate-td47679.html
 * 
 * Eventually, someone should feel free to implement an llvm-specific version
 */

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE rotate(__CLC_GENTYPE x, __CLC_GENTYPE n){
    //Try to avoid extra work if someone's spinning the value through multiple
    //full rotations
    n = n % (__CLC_GENTYPE)__CLC_GENSIZE;

#ifdef __CLC_SCALAR
    if (n > 0){
        return (x << n) | (((__CLC_U_GENTYPE)x) >> (__CLC_GENSIZE - n));
    } else if (n == 0){
        return x;
    } else {
        return ( (((__CLC_U_GENTYPE)x) >> -n) | (x << (__CLC_GENSIZE + n)) );
    }
#else
    //XXX: There's a lot of __builtin_astype calls to cast everything to
    //     unsigned ... This should be improved so that if __CLC_GENTYPE==__CLC_U_GENTYPE, no
    //     casts are required.
    
    __CLC_U_GENTYPE x_1 = __builtin_astype(x, __CLC_U_GENTYPE);

    //XXX: Is (__CLC_U_GENTYPE >> S__CLC_GENTYPE) | (__CLC_U_GENTYPE << S__CLC_GENTYPE) legal?
    //     If so, then combine the amt and shifts into a single set of statements
    
    __CLC_U_GENTYPE amt;
    amt = (n < (__CLC_GENTYPE)0 ? __builtin_astype((__CLC_GENTYPE)0-n, __CLC_U_GENTYPE) : (__CLC_U_GENTYPE)0);
    x_1 = (x_1 >> amt) | (x_1 << ((__CLC_U_GENTYPE)__CLC_GENSIZE - amt));

    amt = (n < (__CLC_GENTYPE)0 ? (__CLC_U_GENTYPE)0 : __builtin_astype(n, __CLC_U_GENTYPE));
    x_1 = (x_1 << amt) | (x_1 >> ((__CLC_U_GENTYPE)__CLC_GENSIZE - amt));

    return __builtin_astype(x_1, __CLC_GENTYPE);
#endif
}
