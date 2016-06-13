
#ifndef OCKL_H
#define OCKL_H

// This C header declares the functions provided by the OCKL library
// Aspects of this library's behavior can be controlled via the 
// oclc library.  See the oclc header for further information

#define _MANGLE3(P,N,S) P##_##N##_##S
#define MANGLE3(P,N,S) _MANGLE3(P,N,S)
#define OCKL_MANGLE_I32(N) MANGLE3(__ockl, N, i32)
#define OCKL_MANGLE_U32(N) MANGLE3(__ockl, N, u32)
#define OCKL_MANGLE_I64(N) MANGLE3(__ockl, N, i64)
#define OCKL_MANGLE_U64(N) MANGLE3(__ockl, N, u64)

#define DECL_OCKL_UNARY_I32(N) extern int OCKL_MANGLE_I32(N)(int);
#define _DECL_X_OCKL_UNARY_I32(A,N) extern __attribute__((A)) int OCKL_MANGLE_I32(N)(int);
#define DECL_PURE_OCKL_UNARY_I32(N) _DECL_X_OCKL_UNARY_I32(pure, N)
#define DECL_CONST_OCKL_UNARY_I32(N) _DECL_X_OCKL_UNARY_I32(const, N)

#define DECL_OCKL_UNARY_I64(N) extern long OCKL_MANGLE_I64(N)(long);
#define _DECL_X_OCKL_UNARY_I64(A,N) extern __attribute__((A)) long OCKL_MANGLE_I64(N)(long);
#define DECL_PURE_OCKL_UNARY_I64(N) _DECL_X_OCKL_UNARY_I64(pure, N)
#define DECL_CONST_OCKL_UNARY_I64(N) _DECL_X_OCKL_UNARY_I64(const, N)

#define DECL_OCKL_UNARY_U32(N) extern uint OCKL_MANGLE_U32(N)(uint);
#define _DECL_X_OCKL_UNARY_U32(A,N) extern __attribute__((A)) uint OCKL_MANGLE_U32(N)(uint);
#define DECL_PURE_OCKL_UNARY_U32(N) _DECL_X_OCKL_UNARY_U32(pure, N)
#define DECL_CONST_OCKL_UNARY_U32(N) _DECL_X_OCKL_UNARY_U32(const, N)

#define DECL_OCKL_UNARY_U64(N) extern ulong OCKL_MANGLE_U64(N)(ulong);
#define _DECL_X_OCKL_UNARY_U64(A,N) extern __attribute__((A)) ulong OCKL_MANGLE_U64(N)(ulong);
#define DECL_PURE_OCKL_UNARY_U64(N) _DECL_X_OCKL_UNARY_U64(pure, N)
#define DECL_CONST_OCKL_UNARY_U64(N) _DECL_X_OCKL_UNARY_U64(const, N)

#define DECL_OCKL_BINARY_I32(N) extern int OCKL_MANGLE_I32(N)(int,int);
#define _DECL_X_OCKL_BINARY_I32(A,N) extern __attribute__((A)) int OCKL_MANGLE_I32(N)(int,int);
#define DECL_PURE_OCKL_BINARY_I32(N) _DECL_X_OCKL_BINARY_I32(pure, N)
#define DECL_CONST_OCKL_BINARY_I32(N) _DECL_X_OCKL_BINARY_I32(const, N)

#define DECL_OCKL_BINARY_I64(N) extern long OCKL_MANGLE_I64(N)(long,long);
#define _DECL_X_OCKL_BINARY_I64(A,N) extern __attribute__((A)) long OCKL_MANGLE_I64(N)(long,long);
#define DECL_PURE_OCKL_BINARY_I64(N) _DECL_X_OCKL_BINARY_I64(pure, N)
#define DECL_CONST_OCKL_BINARY_I64(N) _DECL_X_OCKL_BINARY_I64(const, N)

#define DECL_OCKL_BINARY_U32(N) extern uint OCKL_MANGLE_U32(N)(uint,uint);
#define _DECL_X_OCKL_BINARY_U32(A,N) extern __attribute__((A)) uint OCKL_MANGLE_U32(N)(uint,uint);
#define DECL_PURE_OCKL_BINARY_U32(N) _DECL_X_OCKL_BINARY_U32(pure, N)
#define DECL_CONST_OCKL_BINARY_U32(N) _DECL_X_OCKL_BINARY_U32(const, N)

#define DECL_OCKL_BINARY_U64(N) extern ulong OCKL_MANGLE_U64(N)(ulong,ulong);
#define _DECL_X_OCKL_BINARY_U64(A,N) extern __attribute__((A)) ulong OCKL_MANGLE_U64(N)(ulong,ulong);
#define DECL_PURE_OCKL_BINARY_U64(N) _DECL_X_OCKL_BINARY_U64(pure, N)
#define DECL_CONST_OCKL_BINARY_U64(N) _DECL_X_OCKL_BINARY_U64(const, N)

DECL_CONST_OCKL_UNARY_I32(clz)
DECL_CONST_OCKL_UNARY_I32(ctz)
DECL_CONST_OCKL_UNARY_I32(popcount)

DECL_CONST_OCKL_BINARY_I32(add_sat)
DECL_CONST_OCKL_BINARY_U32(add_sat)
DECL_CONST_OCKL_BINARY_I64(add_sat)
DECL_CONST_OCKL_BINARY_U64(add_sat)

DECL_CONST_OCKL_BINARY_I32(sub_sat)
DECL_CONST_OCKL_BINARY_U32(sub_sat)
DECL_CONST_OCKL_BINARY_I64(sub_sat)
DECL_CONST_OCKL_BINARY_U64(sub_sat)

DECL_CONST_OCKL_BINARY_I32(mul_hi)
DECL_CONST_OCKL_BINARY_U32(mul_hi)
DECL_CONST_OCKL_BINARY_I64(mul_hi)
DECL_CONST_OCKL_BINARY_U64(mul_hi)

DECL_CONST_OCKL_BINARY_I32(mul24)
DECL_CONST_OCKL_BINARY_U32(mul24)

#endif // OCKL_H

