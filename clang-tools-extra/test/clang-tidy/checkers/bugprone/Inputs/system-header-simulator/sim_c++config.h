#ifndef _SIM_CPP_CONFIG_H
#define _SIM_CPP_CONFIG_H

#pragma clang system_header

typedef unsigned char uint8_t;

typedef __typeof__(sizeof(int)) size_t;
typedef __typeof__((char*)0-(char*)0) ptrdiff_t;

#endif // _SIM_CPP_CONFIG_H
