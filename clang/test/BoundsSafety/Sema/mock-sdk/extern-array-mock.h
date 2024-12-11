#ifndef __EXTERN_ARRAY_MOCK_STD_H__
#define __EXTERN_ARRAY_MOCK_STD_H__
extern unsigned externArray[];

inline int f() {
    int *ptr = externArray;
    return *ptr;
}

#endif
