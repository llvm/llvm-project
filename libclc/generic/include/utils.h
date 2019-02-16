#ifndef __CLC_UTILS_H_
#define __CLC_UTILS_H_

#define __CLC_CONCAT(x, y) x ## y
#define __CLC_XCONCAT(x, y) __CLC_CONCAT(x, y)

#define __CLC_STR(x) #x
#define __CLC_XSTR(x) __CLC_STR(x)

#endif
