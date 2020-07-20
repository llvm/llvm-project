// header for codegen-extern-template.cpp
#ifndef CODEGEN_EXTERN_TEMPLATE_H
#define CODEGEN_EXTERN_TEMPLATE_H

template <typename T>
inline T foo() { return 10; }

extern template int foo<int>();

inline int bar() { return foo<int>(); }

#endif
