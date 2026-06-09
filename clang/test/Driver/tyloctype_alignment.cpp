// RUN: %clang -c %s

template<class T>
int fpclassify(T v);
template<>
int fpclassify<float> (float v);
