/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((overloadable, const))

#define char_mask ((char)1 << 7)
#define short_mask ((short)1 << 15)
#define int_mask ((int)1 << 31)
#define long_mask ((long)1 << 63)

#define any_op |
#define all_op &

#define RED(T,O)

#define RED2(T,O) \
    T a = a2.lo O a2.hi

#define RED3(T,O) \
    T a = a3.s0 O a3.s1 O a3.s2

#define RED4(T,O) \
    T##2 a2 = a4.hi O a4.lo; \
    RED2(T,O)

#define RED8(T,O) \
    T##4 a4 = a8.hi O a8.lo; \
    RED4(T,O)

#define RED16(T,O) \
    T##8 a8 = a16.hi O a16.lo; \
    RED8(T,O)

#define RET(T) return (a & T##_mask) != (T)0

#define GENNT(F,N,T) \
ATTR int \
F(T##N a##N) \
{ \
    RED##N(T,F##_op); \
    RET(T); \
}

#define GENT(F,T) \
    GENNT(F,16,T) \
    GENNT(F,8,T) \
    GENNT(F,4,T) \
    GENNT(F,3,T) \
    GENNT(F,2,T) \
    GENNT(F,,T)

#define GEN(F) \
    GENT(F,char) \
    GENT(F,short) \
    GENT(F,int) \
    GENT(F,long)

GEN(any)
GEN(all)
